"""API integration tests (in-process, in-memory repository, short windows)."""

import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.api.settings import Settings
from backend.detection.baseline import DetectorConfig
from backend.models.diagnosis import WorkOrderDraft
from backend.tests.fakes import ScriptedLLM, tool_call, verdict

FAST = Settings(window_duration_s=8.0, sample_rate_hz=4000.0)
FAST_DETECTOR = DetectorConfig(min_baseline_windows=4, consecutive_windows=3)


def spec(asset_id="M-1", **overrides) -> dict:
    base = {
        "asset_id": asset_id,
        "rated_power_kw": 15.0,
        "rated_voltage_v": 460.0,
        "rated_current_a": 22.0,
        "supply_frequency_hz": 60.0,
        "poles": 4,
        "rated_speed_rpm": 1750.0,
        "rated_efficiency": 0.91,
        "locked_rotor_current_ratio": 6.5,
        "rotor_slots": 28,
        "bearings": [
            {"position": "DE", "designation": "6205"},
            {"position": "NDE", "designation": "6203"},
        ],
    }
    base.update(overrides)
    return base


@pytest.fixture
def client():
    with TestClient(create_app(FAST, detector_config=FAST_DETECTOR, llm=None)) as c:
        yield c


def commission(client, asset_id="M-1", **overrides):
    r = client.post("/api/assets", json={"spec": spec(asset_id, **overrides)})
    assert r.status_code == 201, r.text
    return r.json()["data"]


def advance(client, asset_id="M-1", windows=1):
    r = client.post("/api/simulator/advance", json={"asset_id": asset_id, "windows": windows})
    assert r.status_code == 200, r.text
    return r.json()["data"]


class TestCommissioning:
    def test_commission_returns_fault_map_in_envelope(self, client):
        body = client.post("/api/assets", json={"spec": spec()}).json()
        assert body["success"] is True and body["error"] is None
        fmap = body["data"]["commissioning_fault_map"]
        assert fmap["operating_point"]["synchronous_speed_rpm"] == 1800.0
        assert any(b["fault_class"] == "broken_rotor_bar" for b in fmap["bins"])

    def test_invalid_nameplate_rejected_with_physical_reason(self, client):
        r = client.post("/api/assets", json={"spec": spec(poles=5)})
        assert r.status_code == 422
        body = r.json()
        assert body["success"] is False
        assert body["error"]["code"] == "validation_error"
        assert any("even" in d["msg"] for d in body["error"]["details"])

    def test_duplicate_asset_conflict(self, client):
        commission(client)
        r = client.post("/api/assets", json={"spec": spec()})
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "asset_exists"

    def test_incomplete_nameplate_lists_confirmations(self, client):
        data = commission(client, bearings=[], rotor_slots=None, locked_rotor_current_ratio=None)
        params = {c["parameter"] for c in data["needs_confirmation"]}
        assert {"rotor_slots", "locked_rotor_current_ratio"} <= params
        assert "bearing_outer" in data["summary"]["unavailable_fault_classes"]

    def test_commissioning_discloses_baseline_assumption(self, client):
        data = commission(client)
        assert "healthy reference" in data["baseline_assumption"]

    def test_unknown_asset_404_in_envelope(self, client):
        r = client.get("/api/assets/NOPE")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "asset_not_found"


class TestMonitoring:
    def test_list_and_detail_after_windows(self, client):
        commission(client)
        results = advance(client, windows=2)
        assert [r["window_index"] for r in results] == [0, 1]
        listing = client.get("/api/assets").json()
        assert listing["meta"]["total"] == 1
        summary = listing["data"][0]
        assert summary["health"] == "learning"
        assert summary["load_factor"] == pytest.approx(0.8, abs=0.06)
        detail = client.get("/api/assets/M-1").json()["data"]
        assert detail["latest_window"]["slip"]["source"] == "principal_slot_harmonic"

    def test_spectrum_has_bins_overlay_and_is_trimmed(self, client):
        commission(client)
        advance(client)
        view = client.get("/api/assets/M-1/spectrum", params={"max_hz": 200}).json()["data"]
        assert max(view["frequencies_hz"]) <= 200
        assert len(view["frequencies_hz"]) == len(view["level_db"])
        assert any(b["label"] == "BRB lower k=1" for b in view["bins"])

    def test_spectrum_404_before_first_window(self, client):
        commission(client)
        assert client.get("/api/assets/M-1/spectrum").status_code == 404

    def test_injected_fault_raises_single_deduplicated_alert(self, client):
        commission(client)
        advance(client, windows=5)
        r = client.post(
            "/api/simulator/inject",
            json={
                "asset_id": "M-1",
                "faults": [{"fault": "broken_rotor_bar", "severity": 0.4}],
                "advance_windows": 4,
            },
        )
        windows = r.json()["data"]
        assert [len(w["new_alerts"]) > 0 for w in windows] == [False, False, True, False]
        alerts = client.get("/api/assets/M-1/alerts").json()["data"]
        brb = [a for a in alerts if a["fault_class"] == "broken_rotor_bar"]
        assert len(brb) == 1 and brb[0]["windows_seen"] == 2
        evidence = brb[0]["candidate"]["evidence"]
        assert any(e["label"] == "BRB lower k=1" and e["z_score"] > 4 for e in evidence)
        fleet = client.get("/api/assets").json()["data"]
        assert fleet[0]["health"] == "alert"

    def test_fault_needing_missing_bearing_is_domain_error(self, client):
        commission(client, bearings=[])
        r = client.post(
            "/api/simulator/inject",
            json={
                "asset_id": "M-1",
                "faults": [{"fault": "bearing_outer", "severity": 0.5}],
                "advance_windows": 1,
            },
        )
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "domain_error"

    def test_internal_value_error_is_500_not_422(self, monkeypatch):
        app = create_app(FAST, detector_config=FAST_DETECTOR, llm=None)

        def broken(*_args, **_kwargs):
            raise ValueError("expected shape (3, N)")

        monkeypatch.setattr("backend.agents.monitoring.analyze_window", broken)
        with TestClient(app, raise_server_exceptions=False) as c:
            commission(c)
            r = c.post("/api/simulator/advance", json={"asset_id": "M-1", "windows": 1})
        assert r.status_code == 500
        assert "shape" not in r.text

    def test_two_assets_process_concurrently(self, client):
        from concurrent.futures import ThreadPoolExecutor

        commission(client, "A")
        commission(client, "B")
        service = client.app.state.service
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda a: service.advance(a, 2), ["A", "B"]))
        assert {s.asset_id: s.windows_processed for s in service.fleet()} == {"A": 2, "B": 2}

    def test_advance_bounds_validated(self, client):
        commission(client)
        r = client.post("/api/simulator/advance", json={"asset_id": "M-1", "windows": 500})
        assert r.status_code == 422

    def test_diagnose_without_llm_is_503(self, client):
        commission(client)
        r = client.post("/api/assets/M-1/diagnose")
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "agent_unavailable"


DRAFT = WorkOrderDraft(
    confirming_offline_test="Vibration envelope measurement at the DE bearing housing",
    parts=["6205 bearing"],
    recommended_window="Within two weeks",
    actions=["Isolate and lock out", "Replace DE bearing"],
    safety_notes=["Lock out and verify zero energy"],
)


def raise_outer_race_alert(client) -> str:
    commission(client)
    advance(client, windows=5)
    client.post(
        "/api/simulator/inject",
        json={
            "asset_id": "M-1",
            "faults": [{"fault": "bearing_outer", "severity": 0.6, "bearing_position": "DE"}],
            "advance_windows": 3,
        },
    )
    alerts = client.get("/api/assets/M-1/alerts").json()["data"]
    return next(a["id"] for a in alerts if a["fault_class"] == "bearing_outer")


class TestDiagnosis:
    def _client(self, llm):
        return TestClient(create_app(FAST, detector_config=FAST_DETECTOR, llm=llm))

    def test_end_to_end_receipt_with_work_order(self):
        llm = ScriptedLLM(
            [
                tool_call("envelope_analysis", {"bearing_position": "DE"}, "Outer race?"),
                verdict("confirmed", 0.9),
            ],
            parsed=DRAFT,
        )
        with self._client(llm) as c:
            alert_id = raise_outer_race_alert(c)
            r = c.post("/api/assets/M-1/diagnose", json={"alert_id": alert_id})
            assert r.status_code == 200, r.text
            receipt = r.json()["data"]
            alerts = c.get("/api/assets/M-1/alerts").json()["data"]
            health = c.get("/api/assets/M-1").json()["data"]["summary"]["health"]
        assert receipt["verdict"]["status"] == "confirmed"
        assert receipt["verdict"]["evidence"][0]["tool"] == "envelope_analysis"
        assert any(b["equation"].startswith("f_bearing") for b in receipt["predicted_bins"])
        assert receipt["operating_point"]["slip_source"] == "principal_slot_harmonic"
        assert receipt["work_order"]["severity"] in {"high", "critical"}
        assert receipt["limitations"]  # baseline assumption is always disclosed
        stored = next(a for a in alerts if a["id"] == alert_id)
        assert stored["status"] == "confirmed" and stored["receipt"]["alert_id"] == alert_id
        assert health == "alert"

    def test_discarded_alert_is_not_re_raised_while_candidate_persists(self):
        llm = ScriptedLLM(
            [
                tool_call("load_matched_compare", {}),
                verdict("discarded", 0.7, "energy explained by another source"),
            ]
        )
        with self._client(llm) as c:
            alert_id = raise_outer_race_alert(c)
            c.post("/api/assets/M-1/diagnose", json={"alert_id": alert_id})
            windows = advance(c, windows=2)
            alerts = c.get("/api/assets/M-1/alerts").json()["data"]
        assert not any(w["new_alerts"] for w in windows)
        stored = next(a for a in alerts if a["id"] == alert_id)
        assert stored["status"] == "discarded" and stored["receipt"]["verdict"]["discard_reason"]

    def test_discarded_alert_reopens_when_evidence_grows(self):
        llm = ScriptedLLM([tool_call("load_matched_compare", {}), verdict("discarded", 0.7, "x")])
        with self._client(llm) as c:
            alert_id = raise_outer_race_alert(c)
            c.post("/api/assets/M-1/diagnose", json={"alert_id": alert_id})
            service = c.app.state.service
            stored = next(a for a in service.repo.list_alerts("M-1") if a.id == alert_id)
            service.repo.upsert_alert(stored.model_copy(update={"discarded_at_z": 1.0}))
            windows = advance(c, windows=1)
        assert [a["id"] for a in windows[0]["new_alerts"]] == [alert_id]

    def test_diagnose_without_open_alert_is_422(self):
        with self._client(ScriptedLLM()) as c:
            commission(c)
            r = c.post("/api/assets/M-1/diagnose")
        assert r.status_code == 422
        assert "no open alert" in r.json()["error"]["message"]

    def test_commissioning_preview_structured_and_exclusive_inputs(self):
        with self._client(None) as c:
            r = c.post("/api/commissioning/preview", json={"spec": spec(bearings=[])})
            both = c.post(
                "/api/commissioning/preview", json={"spec": spec(), "nameplate_text": "15 kW"}
            )
            text = c.post("/api/commissioning/preview", json={"nameplate_text": "15 kW motor"})
        data = r.json()["data"]
        assert data["status"] == "ready" and data["fault_map"]["bins"]
        assert "bearing_outer" in {u["fault_class"] for u in data["unavailable"]}
        assert both.status_code == 422
        assert text.status_code == 503  # free text needs the LLM


class TestFleet:
    def test_energy_flags_oversized_motor(self, client):
        commission(client, "BIG")
        client.post("/api/simulator/inject", json={"asset_id": "BIG", "advance_windows": 0})
        # re-commission with a light load profile
        client.post(
            "/api/assets",
            json={"spec": spec("LIGHT"), "simulation": {"load_factor": 0.25, "load_jitter": 0.0}},
        )
        advance(client, "BIG")
        advance(client, "LIGHT")
        energy = client.get("/api/fleet/energy").json()["data"]
        assert energy["oversizing_candidates"] == ["LIGHT"]
        light = next(a for a in energy["assets"] if a["asset_id"] == "LIGHT")
        assert light["estimated_waste_kw"] > 0
        assert energy["assets"][0]["asset_id"] == "LIGHT"  # ranked by waste

    def test_websocket_receives_window_telemetry(self, client):
        commission(client)
        with client.websocket_connect("/ws/telemetry") as ws:
            advance(client)
            message = ws.receive_json()
        assert message["type"] == "window"
        assert message["data"]["asset_id"] == "M-1"

    def test_openapi_schema_renders(self, client):
        schema = client.get("/openapi.json").json()
        assert "/api/assets/{asset_id}/spectrum" in schema["paths"]
        assert client.get("/docs").status_code == 200
