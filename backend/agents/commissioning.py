"""Commissioning Agent: nameplate -> validated MotorSpec + FaultMap, and what is unknown.

Structured input never touches the LLM: it is validated and mapped by the physics engine.
Free text (a photo transcription, a datasheet paste) is extracted by the LLM into a schema in
which every field is nullable, so the model can say "not stated" but has no way to invent a
default. Anything the model inferred rather than read is returned as a confirmation item.

The judgement this stage adds is identifying what it does not know:
- missing required nameplate fields -> status needs_input, no spec is produced;
- missing optional fields (bearings, rotor slots, LRC ratio) -> monitoring starts, with the
  affected fault classes listed as unavailable;
- inferred or ambiguous readings -> explicit confirmation items.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from backend.agents.llm import AgentUnavailableError, LLMClient
from backend.models.fault_map import ConfirmationItem, FaultMap, UnavailableFault
from backend.models.motor import MotorSpec
from backend.physics.fault_map import build_fault_map

MAX_TOKENS = 4_000
REQUIRED_FIELDS = (
    "asset_id",
    "rated_power_kw",
    "rated_voltage_v",
    "rated_current_a",
    "supply_frequency_hz",
    "poles",
    "rated_speed_rpm",
)

SYSTEM_PROMPT = """\
Extract an induction-motor nameplate into the schema. Use null for anything the text does not \
state; never fill a value from typical practice. Convert units only by exact arithmetic \
(1 hp = 0.7457 kW) and list every converted or inferred value in uncertain_fields with a short \
note. You may infer the pole count from rated speed and frequency (synchronous speed \
120*f/P just above rated speed) but you must list it in uncertain_fields. Bearing designations \
are copied exactly as written (e.g. 6309-2Z/C3). Rated efficiency is a fraction (0.93, not 93)."""


class ExtractedBearing(BaseModel):
    position: Literal["DE", "NDE"]
    designation: str


class UncertainField(BaseModel):
    field: str
    note: str


class NameplateExtraction(BaseModel):
    asset_id: str | None
    rated_power_kw: float | None
    rated_voltage_v: float | None
    rated_current_a: float | None
    supply_frequency_hz: float | None
    poles: int | None
    rated_speed_rpm: float | None
    rated_efficiency: float | None
    locked_rotor_current_ratio: float | None
    rotor_slots: int | None
    bearings: list[ExtractedBearing]
    uncertain_fields: list[UncertainField]


class CommissioningResult(BaseModel):
    status: Literal["ready", "needs_input", "invalid"]
    spec: MotorSpec | None = None
    fault_map: FaultMap | None = None
    needs_confirmation: list[ConfirmationItem] = Field(default_factory=list)
    unavailable: list[UnavailableFault] = Field(default_factory=list)
    missing_required: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    extracted: dict[str, Any] | None = None
    used_llm: bool = False


def _from_spec_dict(data: dict[str, Any], used_llm: bool) -> CommissioningResult:
    missing = [f for f in REQUIRED_FIELDS if data.get(f) in (None, "")]
    if missing:
        return CommissioningResult(
            status="needs_input", missing_required=missing, extracted=data, used_llm=used_llm
        )
    try:
        spec = MotorSpec.model_validate(data)
    except ValidationError as err:
        return CommissioningResult(
            status="invalid",
            validation_errors=[
                f"{'.'.join(str(p) for p in e['loc']) or 'spec'}: {e['msg']}" for e in err.errors()
            ],
            extracted=data,
            used_llm=used_llm,
        )
    fmap = build_fault_map(spec)
    return CommissioningResult(
        status="ready",
        spec=spec,
        fault_map=fmap,
        needs_confirmation=list(fmap.needs_confirmation),
        unavailable=list(fmap.unavailable),
        extracted=data if used_llm else None,
        used_llm=used_llm,
    )


class CommissioningAgent:
    def __init__(self, llm: LLMClient | None) -> None:
        self.llm = llm

    def from_structured(self, data: dict[str, Any]) -> CommissioningResult:
        return _from_spec_dict(data, used_llm=False)

    def from_text(self, nameplate_text: str, asset_id: str | None = None) -> CommissioningResult:
        if self.llm is None:
            raise AgentUnavailableError(
                "Free-text commissioning needs the LLM; set ANTHROPIC_API_KEY or submit a "
                "structured nameplate"
            )
        response = self.llm.parse(
            output_format=NameplateExtraction,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": nameplate_text}],
        )
        extraction: NameplateExtraction | None = response.parsed_output
        if extraction is None:
            raise AgentUnavailableError("Nameplate extraction returned no structured output")
        data = extraction.model_dump(exclude={"uncertain_fields"})
        if asset_id:
            data["asset_id"] = asset_id
        data = {k: v for k, v in data.items() if v is not None}
        result = _from_spec_dict(data, used_llm=True)
        inferred = [
            ConfirmationItem(
                parameter=u.field,
                reason=f"Read from free text as {data.get(u.field)!r}: {u.note}",
            )
            for u in extraction.uncertain_fields
        ]
        return result.model_copy(
            update={"needs_confirmation": [*inferred, *result.needs_confirmation]}
        )
