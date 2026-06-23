# SmartCAD-AI — FastAPI + React Integration
**Varroc Eureka Challenge 3.0 | Problem Statement 9**

AI-Driven CAD Design Validation with a 3-layer hybrid architecture (Rule Engine → GBM ML → Fusion Engine), served through a FastAPI REST API and a React dashboard.

---

## Project Structure

```
smartcad/
├── backend/
│   ├── engine.py        # Core validation logic (extracted from notebook)
│   ├── main.py          # FastAPI app & REST endpoints
│   ├── requirements.txt
│   └── start.sh         # Quick start script
├── frontend/
│   ├── src/
│   │   ├── App.js       # React dashboard
│   │   └── index.js
│   ├── public/
│   │   └── index.html
│   └── package.json
└── README.md
```

---

## Quick Start

### 1. Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API is live at: **http://localhost:8000**  
Interactive docs: **http://localhost:8000/docs**

### 2. Frontend (React)

```bash
cd frontend
npm install
npm start
```

App is live at: **http://localhost:3000**  
(Proxied to backend on port 8000 via `"proxy"` in package.json)

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Health check (JSON) |
| GET | `/docs` | Swagger UI |
| POST | `/validate` | Run full 3-layer validation |
| GET | `/sample/{product_type}` | Load a preset design |
| GET | `/feature-columns` | List ML feature names |

### POST `/validate` — Request Body

```json
{
  "product_type": "LIGHTING",
  "wall_thickness_mm": 2.5,
  "fillet_radius_mm": 1.0,
  "draft_angle_deg": 2.0,
  "hole_diameter_mm": 8.0,
  "rib_height_mm": 9.0,
  "rib_thickness_mm": 3.0,
  "tolerance_mm": 0.05,
  "material_density": 1.18,
  "surface_finish_ra": 1.2,
  "assembly_clearance_mm": 0.30,
  "overhang_angle_deg": 38.0,
  "min_feature_size_mm": 1.5,
  "aspect_ratio": 3.2,
  "part_weight_kg": 0.45,
  "vent_aspect_ratio": 3.0,
  "cooling_channel_mm": 0.0,
  "ip_groove_mm": 0.0
}
```

### POST `/validate` — Response

```json
{
  "product_type": "LIGHTING",
  "rule_result": {
    "violations": [...],
    "passed_rules": [...],
    "rule_verdict": "FAIL",
    "risk_level": "HIGH",
    "critical_count": 1,
    "major_count": 0,
    "minor_count": 1
  },
  "ml_result": {
    "ml_verdict": "PASS",
    "pass_prob": 62.3,
    "fail_prob": 37.7,
    "confidence": 62.3
  },
  "fusion": {
    "verdict": "FAIL",
    "method": "Rule Override — Critical Varroc Safety Violation",
    "confidence": 99.0
  }
}
```

---

## 3-Layer Architecture

```
CAD Parameters Input (17 features)
         ↓
┌──────────────────────────────┐
│  LAYER 1 — Rule Engine       │
│  12 Varroc-specific rules    │
│  DFM-VAR · LGT-VAR · EV-VAR │
│  → PASS / WARNING / FAIL     │
└────────────┬─────────────────┘
             ↓
┌──────────────────────────────┐
│  LAYER 2 — GBM Model         │
│  Gradient Boosting on 14     │
│  CAD parameters              │
│  → PASS / FAIL + Confidence  │
└────────────┬─────────────────┘
             ↓
┌──────────────────────────────┐
│  LAYER 3 — Fusion Engine     │
│  Priority decision logic     │
│  CRITICAL override · Consensus│
│  → Final verdict + Report    │
└──────────────────────────────┘
```

## Product Families Supported

| Code | Product | Key Rules |
|------|---------|-----------|
| `LIGHTING` | Headlamps, DRL, Tail lamps | Vent AR, Lens clearance, Surface Ra |
| `EV` | PMSM motors, controllers | Cooling channel, Fillet, Aspect ratio |
| `ADAS` | ECU brackets, connectors | IP groove, Feature size |
| `STRUCTURAL` | Brackets, enclosures | Aspect ratio, Overhang, NVH (AIS-056) |

---

## Deployment Notes

For production deployment:
- Set `allow_origins` in CORS to your specific frontend domain
- Use `gunicorn` with `uvicorn.workers.UvicornWorker` for multi-worker serving
- The GBM model is auto-trained and cached to `smartcad_gbm.pkl` on first run
- Set `REACT_APP_API_URL` env var if frontend and backend are on different hosts
