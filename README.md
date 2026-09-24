# Nameplate

Predictive maintenance for VFD-driven induction motors that needs **no additional sensors and no
historical failure data**. Fault detectors are derived from physics at commissioning time: enter
the nameplate, and the system computes exactly which current-spectrum frequency bins to watch on
that machine (Motor Current Signature Analysis).

Every alert is independently verifiable by a technician — it carries the spectrum, the bins that
exceeded baseline, the equation that predicted them, and a plain-language reading.

> Status: under active development (ABB Accelerator 2026 prototype phase).

## Quick start (backend)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r backend/requirements-dev.txt
pytest
```

All backend modules are imported as `backend.<package>` from the repository root. This is
deliberate: `backend/signal/` would otherwise shadow the Python standard-library `signal` module.
