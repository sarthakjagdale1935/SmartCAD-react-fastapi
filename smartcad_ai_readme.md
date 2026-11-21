# SmartCAD-AI

**AI-Driven CAD Design Validation Platform** · *Varroc Eureka Challenge 3.0 (PS9)*

SmartCAD-AI is a full-stack Design for Manufacturability (DFM) validation platform built with **React**, **FastAPI**, and a **3-Layer Hybrid AI** engine. It evaluates 17 geometric, material, and product-specific CAD parameters in real time across four automotive product families (**Lighting**, **EV**, **ADAS**, and **Structural**) to catch tooling and manufacturing defects before production.

---

## Live Deployment

| Service | Platform | Live URL |
| :--- | :--- | :--- |
| **Frontend Web App** | **Vercel** | [https://smart-cad-react-fastapi.vercel.app/](https://smart-cad-react-fastapi.vercel.app/) |
| **Backend REST API** | **Render** | [https://smartcad-react-fastapi.onrender.com](https://smartcad-react-fastapi.onrender.com) |
| **Interactive API Docs** | **Render (Swagger UI)** | [https://smartcad-react-fastapi.onrender.com/docs](https://smartcad-react-fastapi.onrender.com/docs) |

> **Note:** The backend runs on Render's free tier and automatically spins down after 15 minutes of inactivity. The first request after an idle period may take ~30–50 seconds while the server wakes up.

---

## Hybrid AI Architecture

SmartCAD-AI combines deterministic engineering standards with machine learning through a **3-Layer Fusion Pipeline**:

1. **Layer 1 — Deterministic Engineering Rule Engine**
   * Evaluates **14 core CAD parameters** (wall thickness, fillet radius, draft angle, hole diameter, rib height, rib thickness, tolerance, material density, surface finish $R_a$, assembly clearance, overhang angle, minimum feature size, aspect ratio, and part weight) alongside **3 product-specific parameters** (vent aspect ratio, cooling channel width, and IP sealing groove).
   * Classifies standards violations into severity tiers (**CRITICAL**, **MAJOR**, **MINOR**) while tracking all passed engineering checks.
2. **Layer 2 — Gradient Boosting Machine (GBM) Model**
   * Evaluates multi-parameter interactions from historical design patterns to compute **PASS** and **FAIL** probability distributions and model confidence scores.
3. **Layer 3 — Hybrid Decision Fusion**
   * Synthesizes rule-engine compliance and GBM probabilities into a unified final verdict with an overall confidence score:
     * **PASS** — Design satisfies manufacturing rules and ML quality thresholds.
     * **WARNING** — Borderline metrics or minor discrepancies; escalate to Design Lead for manual review.
     * **FAIL** — Critical DFM violations or high failure probability; do not proceed to tooling without rework.

---

## Tech Stack & Cloud Hosting

| Layer | Technologies | Hosting Provider |
| :--- | :--- | :--- |
| **Frontend UI** | React 18, Create React App, Recharts | **Vercel** |
| **Backend API** | Python 3, FastAPI, Pydantic, Uvicorn | **Render** |
| **Validation Core** | Hybrid AI (Rule Engine + GBM Classifier + Fusion) | **Render** |

---

## Repository Structure

```text
SmartCAD-react-fastapi/
└── smartcad/
    ├── README.md
    ├── backend/
    │   ├── engine.py           # 3-layer Hybrid AI (Rule Engine + GBM + Fusion)
    │   ├── main.py             # FastAPI routes, Pydantic schemas & CORS
    │   ├── requirements.txt    # Python dependencies
    │   └── start.sh            # Production startup script
    └── frontend/
        ├── package.json        # React dependencies & build scripts
        ├── public/
        │   └── index.html
        └── src/
            ├── App.js          # Interactive sliders, charts & validation report UI
            └── index.js
```

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | API status and documentation pointer |
| `GET` | `/health` | Service health check (`{"status": "ok"}`) |
| `GET` | `/feature-columns` | Returns the ordered list of CAD feature columns |
| `GET` | `/sample/{product_type}` | Loads preset CAD parameters (`LIGHTING`, `EV`, `ADAS`, `STRUCTURAL`) |
| `POST` | `/validate` | Runs 3-layer Hybrid AI validation on a CAD parameter payload |

---

## Cloud Deployment (Render + Vercel)

### 1. Backend Deployment on Render
1. Create a **New Web Service** on [Render](https://render.com) connected to this GitHub repository.
2. Set **Root Directory** to `smartcad/backend`.
3. Set **Build Command** to `pip install -r requirements.txt`.
4. Set **Start Command** to `uvicorn main:app --host 0.0.0.0 --port $PORT`.
5. Live service URL: `https://smartcad-react-fastapi.onrender.com`.

### 2. Frontend Deployment on Vercel
1. Import this GitHub repository into [Vercel](https://vercel.com).
2. Set **Root Directory** to `smartcad/frontend` and **Framework Preset** to `Create React App`.
3. Add the environment variable:
   * `REACT_APP_API_URL` = `https://smartcad-react-fastapi.onrender.com`
4. Live frontend URL: `https://smart-cad-react-fastapi.vercel.app/`.

---

## Local Development

```bash
# 1. Start the FastAPI backend
cd smartcad/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# 2. Start the React frontend (in a second terminal)
cd smartcad/frontend
npm install
npm start
```