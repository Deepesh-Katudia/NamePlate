"""Rolling-element bearing geometry database and characteristic defect frequencies.

Provenance matters here. Manufacturers rarely publish internal geometry, so each entry is
tagged:

- PUBLISHED_REFERENCE: geometry reproduces a published defect-frequency table
  (Case Western Reserve University Bearing Data Center, SKF 6203/6205 JEM).
- NOMINAL: pitch diameter from (bore + OD) / 2 with catalogue-typical ball complement.
  Good to a few percent, which is inside the bin search tolerance, but it varies by maker
  and cage type, so the fault map asks a human to confirm it against the vendor's sheet.

An unrecognised designation raises `UnknownBearingError`. There is deliberately no
"average bearing" fallback: a wrong geometry puts the search bins in the wrong place and the
system would confidently report a healthy bearing it never looked at.
"""

from __future__ import annotations

import math
import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from backend.models.motor import BearingGeometry

REQUIRED_GEOMETRY_PARAMETERS: tuple[str, ...] = (
    "ball_count",
    "ball_diameter_mm",
    "pitch_diameter_mm",
    "contact_angle_rad",
)

_INCH_MM = 25.4
_DESIGNATION_PATTERN = re.compile(r"^(NU|NJ|NUP|N)?\d{3,5}")


class GeometryProvenance(StrEnum):
    PUBLISHED_REFERENCE = "published_reference"
    NOMINAL = "nominal"
    USER_SUPPLIED = "user_supplied"


class UnknownBearingError(LookupError):
    """Raised when a bearing designation is not in the database.

    Carries the parameters the operator must supply so callers can surface a precise
    "needs human confirmation" item instead of guessing.
    """

    def __init__(self, designation: str) -> None:
        self.designation = designation
        self.required_parameters = REQUIRED_GEOMETRY_PARAMETERS
        super().__init__(
            f"Unknown bearing designation '{designation}'. Supply its geometry explicitly: "
            + ", ".join(self.required_parameters)
        )


class BearingRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    designation: str
    geometry: BearingGeometry
    provenance: GeometryProvenance
    note: str


class BearingCharacteristicFrequencies(BaseModel):
    """Mechanical defect frequencies in Hz at a given shaft speed."""

    model_config = ConfigDict(frozen=True)

    bpfo: float
    bpfi: float
    bsf: float
    ftf: float

    def as_dict(self) -> dict[str, float]:
        return {"BPFO": self.bpfo, "BPFI": self.bpfi, "BSF": self.bsf, "FTF": self.ftf}


def _geom(n: int, bd: float, pd: float, phi: float = 0.0) -> BearingGeometry:
    return BearingGeometry(
        ball_count=n, ball_diameter_mm=bd, pitch_diameter_mm=pd, contact_angle_rad=phi
    )


_CWRU_NOTE = "CWRU Bearing Data Center, SKF JEM geometry; reproduces published defect table"
_NOMINAL_NOTE = "Nominal: P_d = (bore + OD)/2, catalogue-typical ball set; confirm with vendor"

BEARING_DATABASE: dict[str, BearingRecord] = {
    r.designation: r
    for r in (
        BearingRecord(
            designation="6203",
            geometry=_geom(8, 0.2656 * _INCH_MM, 1.122 * _INCH_MM),
            provenance=GeometryProvenance.PUBLISHED_REFERENCE,
            note=_CWRU_NOTE,
        ),
        BearingRecord(
            designation="6205",
            geometry=_geom(9, 0.3126 * _INCH_MM, 1.537 * _INCH_MM),
            provenance=GeometryProvenance.PUBLISHED_REFERENCE,
            note=_CWRU_NOTE,
        ),
        *(
            BearingRecord(
                designation=d,
                geometry=_geom(n, bd, pd),
                provenance=GeometryProvenance.NOMINAL,
                note=_NOMINAL_NOTE,
            )
            for d, n, bd, pd in (
                # designation, balls, ball dia (mm), pitch dia (mm)
                ("6204", 8, 7.938, 33.5),
                ("6206", 9, 9.525, 46.0),
                ("6207", 9, 11.112, 53.5),
                ("6208", 9, 12.700, 60.0),
                ("6305", 8, 10.319, 43.5),
                ("6306", 8, 11.906, 51.0),
                ("6309", 8, 17.462, 72.5),
                ("6310", 8, 19.050, 80.0),
                ("NU210", 14, 11.000, 70.0),
                ("NU310", 13, 15.000, 80.0),
            )
        ),
    )
}


def normalise_designation(designation: str) -> str | None:
    """Strip seal/clearance suffixes: '6205-2RS C3' -> '6205', 'nu210 ecp' -> 'NU210'."""
    match = _DESIGNATION_PATTERN.match(designation.strip().upper())
    return match.group(0) if match else None


def lookup_bearing(designation: str) -> BearingRecord:
    """Return geometry for a designation, or raise `UnknownBearingError`. Never defaults."""
    key = normalise_designation(designation)
    if key is None or key not in BEARING_DATABASE:
        raise UnknownBearingError(designation)
    return BEARING_DATABASE[key]


def bearing_characteristic_frequencies(
    geometry: BearingGeometry, rotor_freq_hz: float
) -> BearingCharacteristicFrequencies:
    """BPFO, BPFI, BSF, FTF for a stationary outer race and rotating inner race.

    BPFO = (N_b/2) f_r (1 - (B_d/P_d) cos phi)
    BPFI = (N_b/2) f_r (1 + (B_d/P_d) cos phi)
    BSF  = (P_d/(2 B_d)) f_r (1 - ((B_d/P_d) cos phi)^2)
    FTF  = (f_r/2) (1 - (B_d/P_d) cos phi)
    """
    if rotor_freq_hz <= 0:
        raise ValueError(f"rotor_freq_hz must be positive (got {rotor_freq_hz})")
    ratio = geometry.ball_diameter_mm / geometry.pitch_diameter_mm
    ratio_cos = ratio * math.cos(geometry.contact_angle_rad)
    half_n = geometry.ball_count / 2.0
    return BearingCharacteristicFrequencies(
        bpfo=half_n * rotor_freq_hz * (1 - ratio_cos),
        bpfi=half_n * rotor_freq_hz * (1 + ratio_cos),
        bsf=(1 / (2 * ratio)) * rotor_freq_hz * (1 - ratio_cos**2),
        ftf=(rotor_freq_hz / 2) * (1 - ratio_cos),
    )
