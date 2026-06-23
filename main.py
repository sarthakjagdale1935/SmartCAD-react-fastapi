"""
SmartCAD-AI — FastAPI Backend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Literal
from engine import validate_design, FEATURE_COLUMNS

app = FastAPI(
    title="SmartCAD-AI API",
    description="Varroc Eureka Challenge 3.0 | PS9 — AI-Driven CAD Design Validation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class DesignInput(BaseModel):
    product_type: Literal["LIGHTING", "EV", "ADAS", "STRUCTURAL"]
    wall_thickness_mm:     float = Field(..., ge=0.1,   le=10.0)
    fillet_radius_mm:      float = Field(..., ge=0.1,   le=5.0)
    draft_angle_deg:       float = Field(..., ge=0.0,   le=10.0)
    hole_diameter_mm:      float = Field(..., ge=0.5,   le=50.0)
    rib_height_mm:         float = Field(..., ge=1.0,   le=50.0)
    rib_thickness_mm:      float = Field(..., ge=0.5,   le=10.0)
    tolerance_mm:          float = Field(..., ge=0.001, le=0.5)
    material_density:      float = Field(..., ge=0.5,   le=8.0)
    surface_finish_ra:     float = Field(..., ge=0.1,   le=10.0)
    assembly_clearance_mm: float = Field(..., ge=0.01,  le=2.0)
    overhang_angle_deg:    float = Field(..., ge=0.0,   le=90.0)
    min_feature_size_mm:   float = Field(..., ge=0.1,   le=10.0)
    aspect_ratio:          float = Field(..., ge=1.0,   le=20.0)
    part_weight_kg:        float = Field(..., ge=0.01,  le=20.0)
    vent_aspect_ratio:     float = Field(0.0, ge=0.0,   le=20.0)
    cooling_channel_mm:    float = Field(0.0, ge=0.0,   le=20.0)
    ip_groove_mm:          float = Field(0.0, ge=0.0,   le=10.0)


@app.get("/")
def root():
    return {"status": "SmartCAD-AI API running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/feature-columns")
def feature_columns():
    return {"features": FEATURE_COLUMNS}

@app.post("/validate")
def validate(body: DesignInput):
    design = body.model_dump()
    product_type = design.pop("product_type")
    try:
        return validate_design(design, product_type=product_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sample/{product_type}")
def sample(product_type: Literal["LIGHTING", "EV", "ADAS", "STRUCTURAL"]):
    samples = {
        "LIGHTING": {"product_type":"LIGHTING","wall_thickness_mm":2.1,"fillet_radius_mm":1.1,"draft_angle_deg":2.0,"hole_diameter_mm":8.0,"rib_height_mm":9.0,"rib_thickness_mm":3.0,"tolerance_mm":0.05,"material_density":1.18,"surface_finish_ra":1.2,"assembly_clearance_mm":0.12,"overhang_angle_deg":38.0,"min_feature_size_mm":1.5,"aspect_ratio":3.2,"part_weight_kg":0.45,"vent_aspect_ratio":3.0,"cooling_channel_mm":0.0,"ip_groove_mm":0.0},
        "EV":       {"product_type":"EV","wall_thickness_mm":1.4,"fillet_radius_mm":0.7,"draft_angle_deg":1.8,"hole_diameter_mm":11.0,"rib_height_mm":9.5,"rib_thickness_mm":3.8,"tolerance_mm":0.045,"material_density":2.71,"surface_finish_ra":1.1,"assembly_clearance_mm":0.28,"overhang_angle_deg":41.0,"min_feature_size_mm":1.3,"aspect_ratio":4.0,"part_weight_kg":1.95,"vent_aspect_ratio":0.0,"cooling_channel_mm":2.1,"ip_groove_mm":1.4},
        "ADAS":     {"product_type":"ADAS","wall_thickness_mm":2.6,"fillet_radius_mm":1.0,"draft_angle_deg":2.1,"hole_diameter_mm":7.0,"rib_height_mm":7.2,"rib_thickness_mm":3.2,"tolerance_mm":0.052,"material_density":1.05,"surface_finish_ra":1.35,"assembly_clearance_mm":0.31,"overhang_angle_deg":33.0,"min_feature_size_mm":0.98,"aspect_ratio":2.7,"part_weight_kg":0.20,"vent_aspect_ratio":0.0,"cooling_channel_mm":0.0,"ip_groove_mm":1.5},
        "STRUCTURAL":{"product_type":"STRUCTURAL","wall_thickness_mm":3.5,"fillet_radius_mm":2.0,"draft_angle_deg":3.0,"hole_diameter_mm":15.0,"rib_height_mm":12.0,"rib_thickness_mm":4.8,"tolerance_mm":0.08,"material_density":2.76,"surface_finish_ra":2.0,"assembly_clearance_mm":0.40,"overhang_angle_deg":40.0,"min_feature_size_mm":2.2,"aspect_ratio":4.0,"part_weight_kg":2.25,"vent_aspect_ratio":0.0,"cooling_channel_mm":0.0,"ip_groove_mm":0.0},
    }
    return samples[product_type]
