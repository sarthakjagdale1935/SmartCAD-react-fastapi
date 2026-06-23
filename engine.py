"""
SmartCAD-AI — Core Validation Engine
Extracted from SmartCAD_AI.ipynb (Varroc Eureka Challenge 3.0 | PS9)
"""

import numpy as np
import pickle
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Feature columns (14 ML features) ─────────────────────────────────────────
FEATURE_COLUMNS = [
    'wall_thickness_mm', 'fillet_radius_mm', 'draft_angle_deg',
    'hole_diameter_mm', 'rib_height_mm', 'rib_thickness_mm',
    'tolerance_mm', 'material_density', 'surface_finish_ra',
    'assembly_clearance_mm', 'overhang_angle_deg', 'min_feature_size_mm',
    'aspect_ratio', 'part_weight_kg',
]

# ── Training dataset (54 designs — from notebook) ────────────────────────────
TRAINING_DATA = [
    # LIGHTING (16)
    {'product_type':'LIGHTING','label':'PASS','wall_thickness_mm':2.5,'fillet_radius_mm':1.2,'draft_angle_deg':2.0,'hole_diameter_mm':8.0,'rib_height_mm':9.0,'rib_thickness_mm':3.5,'tolerance_mm':0.05,'material_density':1.18,'surface_finish_ra':1.2,'assembly_clearance_mm':0.30,'overhang_angle_deg':35.0,'min_feature_size_mm':1.5,'aspect_ratio':3.2,'part_weight_kg':0.45,'vent_aspect_ratio':3.0,'cooling_channel_mm':0.0,'ip_groove_mm':0.0},
    {'product_type':'LIGHTING','label':'FAIL','wall_thickness_mm':1.4,'fillet_radius_mm':0.6,'draft_angle_deg':0.8,'hole_diameter_mm':5.0,'rib_height_mm':14.0,'rib_thickness_mm':2.0,'tolerance_mm':0.12,'material_density':1.05,'surface_finish_ra':2.5,'assembly_clearance_mm':0.10,'overhang_angle_deg':55.0,'min_feature_size_mm':0.5,'aspect_ratio':7.0,'part_weight_kg':0.30,'vent_aspect_ratio':5.0,'cooling_channel_mm':0.0,'ip_groove_mm':0.0},
    {'product_type':'LIGHTING','label':'PASS','wall_thickness_mm':3.0,'fillet_radius_mm':1.5,'draft_angle_deg':2.5,'hole_diameter_mm':10.0,'rib_height_mm':8.0,'rib_thickness_mm':3.0,'tolerance_mm':0.04,'material_density':1.20,'surface_finish_ra':1.0,'assembly_clearance_mm':0.35,'overhang_angle_deg':30.0,'min_feature_size_mm':2.0,'aspect_ratio':2.5,'part_weight_kg':0.55,'vent_aspect_ratio':2.5,'cooling_channel_mm':0.0,'ip_groove_mm':0.0},
    {'product_type':'LIGHTING','label':'PASS','wall_thickness_mm':2.2,'fillet_radius_mm':1.0,'draft_angle_deg':1.8,'hole_diameter_mm':7.0,'rib_height_mm':10.0,'rib_thickness_mm':3.5,'tolerance_mm':0.06,'material_density':1.15,'surface_finish_ra':1.4,'assembly_clearance_mm':0.25,'overhang_angle_deg':38.0,'min_feature_size_mm':1.2,'aspect_ratio':3.5,'part_weight_kg':0.40,'vent_aspect_ratio':3.5,'cooling_channel_mm':0.0,'ip_groove_mm':0.0},
    {'product_type':'LIGHTING','label':'FAIL','wall_thickness_mm':1.2,'fillet_radius_mm':0.4,'draft_angle_deg':0.6,'hole_diameter_mm':3.5,'rib_height_mm':20.0,'rib_thickness_mm':2.5,'tolerance_mm':0.15,'material_density':1.10,'surface_finish_ra':3.5,'assembly_clearance_mm':0.06,'overhang_angle_deg':60.0,'min_feature_size_mm':0.3,'aspect_ratio':9.0,'part_weight_kg':0.25,'vent_aspect_ratio':6.0,'cooling_channel_mm':0.0,'ip_groove_mm':0.0},
    {'product_type':'LIGHTING','label':'PASS','wall_thickness_mm':2.8,'fillet_radius_mm':1.3,'draft_angle_deg':2.2,'hole_diameter_mm':9.0,'rib_height_mm':9.5,'rib_thickness_mm':3.2,'tolerance_mm':0.045,'material_density':1.22,'surface_finish_ra':1.1,'assembly_clearance_mm':0.28,'overhang_angle_deg':32.0,'min_feature_size_mm':1.8,'aspect_ratio':2.8,'part_weight_kg':0.50,'vent_aspect_ratio':2.8,'cooling_channel_mm':0.0,'ip_groove_mm':0.0},
    {'product_type':'LIGHTING','label':'FAIL','wall_thickness_mm':1.6,'fillet_radius_mm':0.5,'draft_angle_deg':1.0,'hole_diameter_mm':4.0,'rib_height_mm':18.0,'rib_thickness_mm':2.2,'tolerance_mm':0.13,'material_density':1.08,'surface_finish_ra':3.0,'assembly_clearance_mm':0.08,'overhang_angle_deg':58.0,'min_feature_size_mm':0.4,'aspect_ratio':8.0,'part_weight_kg':0.28,'vent_aspect_ratio':5.5,'cooling_channel_mm':0.0,'ip_groove_mm':0.0},
    {'product_type':'LIGHTING','label':'PASS','wall_thickness_mm':2.6,'fillet_radius_mm':1.1,'draft_angle_deg':2.0,'hole_diameter_mm':8.5,'rib_height_mm':9.2,'rib_thickness_mm':3.4,'tolerance_mm':0.05,'material_density':1.19,'surface_finish_ra':1.3,'assembly_clearance_mm':0.32,'overhang_angle_deg':34.0,'min_feature_size_mm':1.6,'aspect_ratio':3.0,'part_weight_kg':0.48,'vent_aspect_ratio':3.2,'cooling_channel_mm':0.0,'ip_groove_mm':0.0},
    {'product_type':'LIGHTING','label':'FAIL','wall_thickness_mm':1.3,'fillet_radius_mm':0.45,'draft_angle_deg':0.7,'hole_diameter_mm':3.8,'rib_height_mm':19.0,'rib_thickness_mm':2.3,'tolerance_mm':0.14,'material_density':1.07,'surface_finish_ra':3.8,'assembly_clearance_mm':0.07,'overhang_angle_deg':59.0,'min_feature_size_mm':0.35,'aspect_ratio':8.5,'part_weight_kg':0.27,'vent_aspect_ratio':5.8,'cooling_channel_mm':0.0,'ip_groove_mm':0.0},
    {'product_type':'LIGHTING','label':'PASS','wall_thickness_mm':2.4,'fillet_radius_mm':1.05,'draft_angle_deg':1.9,'hole_diameter_mm':7.5,'rib_height_mm':9.8,'rib_thickness_mm':3.3,'tolerance_mm':0.055,'material_density':1.17,'surface_finish_ra':1.25,'assembly_clearance_mm':0.27,'overhang_angle_deg':36.0,'min_feature_size_mm':1.4,'aspect_ratio':3.3,'part_weight_kg':0.44,'vent_aspect_ratio':3.3,'cooling_channel_mm':0.0,'ip_groove_mm':0.0},
    {'product_type':'LIGHTING','label':'FAIL','wall_thickness_mm':1.5,'fillet_radius_mm':0.55,'draft_angle_deg':0.9,'hole_diameter_mm':4.5,'rib_height_mm':16.0,'rib_thickness_mm':2.1,'tolerance_mm':0.11,'material_density':1.09,'surface_finish_ra':2.8,'assembly_clearance_mm':0.09,'overhang_angle_deg':56.0,'min_feature_size_mm':0.45,'aspect_ratio':7.5,'part_weight_kg':0.29,'vent_aspect_ratio':5.2,'cooling_channel_mm':0.0,'ip_groove_mm':0.0},
    {'product_type':'LIGHTING','label':'PASS','wall_thickness_mm':3.0,'fillet_radius_mm':1.4,'draft_angle_deg':2.3,'hole_diameter_mm':9.5,'rib_height_mm':8.5,'rib_thickness_mm':3.8,'tolerance_mm':0.04,'material_density':1.21,'surface_finish_ra':1.05,'assembly_clearance_mm':0.38,'overhang_angle_deg':31.0,'min_feature_size_mm':1.9,'aspect_ratio':2.6,'part_weight_kg':0.52,'vent_aspect_ratio':2.6,'cooling_channel_mm':0.0,'ip_groove_mm':0.0},
    {'product_type':'LIGHTING','label':'FAIL','wall_thickness_mm':1.4,'fillet_radius_mm':0.5,'draft_angle_deg':0.8,'hole_diameter_mm':4.2,'rib_height_mm':17.0,'rib_thickness_mm':2.0,'tolerance_mm':0.12,'material_density':1.06,'surface_finish_ra':3.2,'assembly_clearance_mm':0.08,'overhang_angle_deg':57.0,'min_feature_size_mm':0.42,'aspect_ratio':7.8,'part_weight_kg':0.26,'vent_aspect_ratio':5.6,'cooling_channel_mm':0.0,'ip_groove_mm':0.0},
    {'product_type':'LIGHTING','label':'PASS','wall_thickness_mm':2.3,'fillet_radius_mm':1.02,'draft_angle_deg':1.85,'hole_diameter_mm':7.2,'rib_height_mm':9.6,'rib_thickness_mm':3.45,'tolerance_mm':0.052,'material_density':1.16,'surface_finish_ra':1.28,'assembly_clearance_mm':0.26,'overhang_angle_deg':37.0,'min_feature_size_mm':1.3,'aspect_ratio':3.4,'part_weight_kg':0.43,'vent_aspect_ratio':3.4,'cooling_channel_mm':0.0,'ip_groove_mm':0.0},
    # EV (14)
    {'product_type':'EV','label':'PASS','wall_thickness_mm':2.5,'fillet_radius_mm':1.0,'draft_angle_deg':2.0,'hole_diameter_mm':12.0,'rib_height_mm':10.0,'rib_thickness_mm':4.0,'tolerance_mm':0.04,'material_density':2.71,'surface_finish_ra':1.5,'assembly_clearance_mm':0.35,'overhang_angle_deg':35.0,'min_feature_size_mm':1.5,'aspect_ratio':3.0,'part_weight_kg':1.80,'vent_aspect_ratio':0.0,'cooling_channel_mm':4.0,'ip_groove_mm':1.5},
    {'product_type':'EV','label':'FAIL','wall_thickness_mm':1.4,'fillet_radius_mm':0.7,'draft_angle_deg':1.8,'hole_diameter_mm':11.0,'rib_height_mm':9.5,'rib_thickness_mm':3.8,'tolerance_mm':0.045,'material_density':2.71,'surface_finish_ra':1.1,'assembly_clearance_mm':0.28,'overhang_angle_deg':41.0,'min_feature_size_mm':1.3,'aspect_ratio':4.0,'part_weight_kg':1.95,'vent_aspect_ratio':0.0,'cooling_channel_mm':2.1,'ip_groove_mm':1.4},
    {'product_type':'EV','label':'PASS','wall_thickness_mm':3.0,'fillet_radius_mm':1.2,'draft_angle_deg':2.5,'hole_diameter_mm':15.0,'rib_height_mm':8.0,'rib_thickness_mm':4.5,'tolerance_mm':0.035,'material_density':2.65,'surface_finish_ra':1.3,'assembly_clearance_mm':0.40,'overhang_angle_deg':30.0,'min_feature_size_mm':2.0,'aspect_ratio':2.5,'part_weight_kg':2.20,'vent_aspect_ratio':0.0,'cooling_channel_mm':5.0,'ip_groove_mm':1.8},
    {'product_type':'EV','label':'PASS','wall_thickness_mm':2.2,'fillet_radius_mm':0.9,'draft_angle_deg':1.9,'hole_diameter_mm':11.5,'rib_height_mm':9.0,'rib_thickness_mm':3.8,'tolerance_mm':0.042,'material_density':2.70,'surface_finish_ra':1.4,'assembly_clearance_mm':0.32,'overhang_angle_deg':36.0,'min_feature_size_mm':1.6,'aspect_ratio':3.2,'part_weight_kg':1.90,'vent_aspect_ratio':0.0,'cooling_channel_mm':4.5,'ip_groove_mm':1.6},
    {'product_type':'EV','label':'FAIL','wall_thickness_mm':1.2,'fillet_radius_mm':0.5,'draft_angle_deg':1.2,'hole_diameter_mm':8.0,'rib_height_mm':18.0,'rib_thickness_mm':3.0,'tolerance_mm':0.08,'material_density':2.75,'surface_finish_ra':1.8,'assembly_clearance_mm':0.15,'overhang_angle_deg':50.0,'min_feature_size_mm':0.8,'aspect_ratio':6.0,'part_weight_kg':1.50,'vent_aspect_ratio':0.0,'cooling_channel_mm':1.5,'ip_groove_mm':0.9},
    {'product_type':'EV','label':'PASS','wall_thickness_mm':2.8,'fillet_radius_mm':1.1,'draft_angle_deg':2.2,'hole_diameter_mm':13.0,'rib_height_mm':8.5,'rib_thickness_mm':4.2,'tolerance_mm':0.038,'material_density':2.68,'surface_finish_ra':1.2,'assembly_clearance_mm':0.38,'overhang_angle_deg':32.0,'min_feature_size_mm':1.8,'aspect_ratio':2.8,'part_weight_kg':2.0,'vent_aspect_ratio':0.0,'cooling_channel_mm':4.8,'ip_groove_mm':1.7},
    {'product_type':'EV','label':'FAIL','wall_thickness_mm':1.5,'fillet_radius_mm':0.6,'draft_angle_deg':1.4,'hole_diameter_mm':9.0,'rib_height_mm':16.0,'rib_thickness_mm':3.2,'tolerance_mm':0.07,'material_density':2.72,'surface_finish_ra':1.7,'assembly_clearance_mm':0.18,'overhang_angle_deg':48.0,'min_feature_size_mm':0.9,'aspect_ratio':5.5,'part_weight_kg':1.60,'vent_aspect_ratio':0.0,'cooling_channel_mm':1.8,'ip_groove_mm':1.0},
    {'product_type':'EV','label':'PASS','wall_thickness_mm':2.6,'fillet_radius_mm':1.05,'draft_angle_deg':2.1,'hole_diameter_mm':12.5,'rib_height_mm':9.2,'rib_thickness_mm':4.1,'tolerance_mm':0.04,'material_density':2.69,'surface_finish_ra':1.35,'assembly_clearance_mm':0.36,'overhang_angle_deg':33.0,'min_feature_size_mm':1.7,'aspect_ratio':2.9,'part_weight_kg':1.95,'vent_aspect_ratio':0.0,'cooling_channel_mm':4.6,'ip_groove_mm':1.6},
    {'product_type':'EV','label':'FAIL','wall_thickness_mm':1.3,'fillet_radius_mm':0.55,'draft_angle_deg':1.3,'hole_diameter_mm':8.5,'rib_height_mm':17.0,'rib_thickness_mm':3.1,'tolerance_mm':0.075,'material_density':2.73,'surface_finish_ra':1.75,'assembly_clearance_mm':0.16,'overhang_angle_deg':49.0,'min_feature_size_mm':0.85,'aspect_ratio':5.8,'part_weight_kg':1.55,'vent_aspect_ratio':0.0,'cooling_channel_mm':1.6,'ip_groove_mm':0.95},
    {'product_type':'EV','label':'PASS','wall_thickness_mm':2.4,'fillet_radius_mm':0.95,'draft_angle_deg':2.0,'hole_diameter_mm':12.0,'rib_height_mm':9.5,'rib_thickness_mm':3.9,'tolerance_mm':0.041,'material_density':2.70,'surface_finish_ra':1.38,'assembly_clearance_mm':0.34,'overhang_angle_deg':34.0,'min_feature_size_mm':1.65,'aspect_ratio':3.1,'part_weight_kg':1.88,'vent_aspect_ratio':0.0,'cooling_channel_mm':4.4,'ip_groove_mm':1.55},
    {'product_type':'EV','label':'PASS','wall_thickness_mm':2.9,'fillet_radius_mm':1.15,'draft_angle_deg':2.3,'hole_diameter_mm':14.0,'rib_height_mm':8.2,'rib_thickness_mm':4.3,'tolerance_mm':0.037,'material_density':2.66,'surface_finish_ra':1.25,'assembly_clearance_mm':0.39,'overhang_angle_deg':31.0,'min_feature_size_mm':1.85,'aspect_ratio':2.7,'part_weight_kg':2.10,'vent_aspect_ratio':0.0,'cooling_channel_mm':4.9,'ip_groove_mm':1.75},
    {'product_type':'EV','label':'FAIL','wall_thickness_mm':1.4,'fillet_radius_mm':0.62,'draft_angle_deg':1.35,'hole_diameter_mm':8.8,'rib_height_mm':15.5,'rib_thickness_mm':3.15,'tolerance_mm':0.072,'material_density':2.74,'surface_finish_ra':1.72,'assembly_clearance_mm':0.17,'overhang_angle_deg':47.0,'min_feature_size_mm':0.88,'aspect_ratio':5.6,'part_weight_kg':1.58,'vent_aspect_ratio':0.0,'cooling_channel_mm':1.7,'ip_groove_mm':0.98},
    {'product_type':'EV','label':'PASS','wall_thickness_mm':2.7,'fillet_radius_mm':1.08,'draft_angle_deg':2.15,'hole_diameter_mm':13.5,'rib_height_mm':8.8,'rib_thickness_mm':4.15,'tolerance_mm':0.039,'material_density':2.67,'surface_finish_ra':1.28,'assembly_clearance_mm':0.37,'overhang_angle_deg':33.5,'min_feature_size_mm':1.75,'aspect_ratio':2.85,'part_weight_kg':2.05,'vent_aspect_ratio':0.0,'cooling_channel_mm':4.7,'ip_groove_mm':1.65},
    {'product_type':'EV','label':'FAIL','wall_thickness_mm':1.6,'fillet_radius_mm':0.65,'draft_angle_deg':1.45,'hole_diameter_mm':9.5,'rib_height_mm':15.0,'rib_thickness_mm':3.25,'tolerance_mm':0.068,'material_density':2.71,'surface_finish_ra':1.65,'assembly_clearance_mm':0.19,'overhang_angle_deg':46.0,'min_feature_size_mm':0.92,'aspect_ratio':5.3,'part_weight_kg':1.62,'vent_aspect_ratio':0.0,'cooling_channel_mm':1.9,'ip_groove_mm':1.02},
    # ADAS (12)
    {'product_type':'ADAS','label':'PASS','wall_thickness_mm':2.6,'fillet_radius_mm':1.0,'draft_angle_deg':2.1,'hole_diameter_mm':7.0,'rib_height_mm':7.2,'rib_thickness_mm':3.2,'tolerance_mm':0.052,'material_density':1.05,'surface_finish_ra':1.35,'assembly_clearance_mm':0.31,'overhang_angle_deg':33.0,'min_feature_size_mm':0.98,'aspect_ratio':2.7,'part_weight_kg':0.20,'vent_aspect_ratio':0.0,'cooling_channel_mm':0.0,'ip_groove_mm':1.5},
    {'product_type':'ADAS','label':'FAIL','wall_thickness_mm':1.5,'fillet_radius_mm':0.6,'draft_angle_deg':1.1,'hole_diameter_mm':4.5,'rib_height_mm':15.0,'rib_thickness_mm':2.5,'tolerance_mm':0.10,'material_density':1.08,'surface_finish_ra':2.2,'assembly_clearance_mm':0.12,'overhang_angle_deg':52.0,'min_feature_size_mm':0.45,'aspect_ratio':6.5,'part_weight_kg':0.28,'vent_aspect_ratio':0.0,'cooling_channel_mm':0.0,'ip_groove_mm':0.8},
    {'product_type':'ADAS','label':'PASS','wall_thickness_mm':3.0,'fillet_radius_mm':1.2,'draft_angle_deg':2.4,'hole_diameter_mm':9.0,'rib_height_mm':6.5,'rib_thickness_mm':3.5,'tolerance_mm':0.04,'material_density':1.04,'surface_finish_ra':1.1,'assembly_clearance_mm':0.38,'overhang_angle_deg':28.0,'min_feature_size_mm':1.5,'aspect_ratio':2.2,'part_weight_kg':0.22,'vent_aspect_ratio':0.0,'cooling_channel_mm':0.0,'ip_groove_mm':1.8},
    {'product_type':'ADAS','label':'PASS','wall_thickness_mm':2.4,'fillet_radius_mm':0.95,'draft_angle_deg':1.95,'hole_diameter_mm':6.8,'rib_height_mm':7.5,'rib_thickness_mm':3.1,'tolerance_mm':0.055,'material_density':1.06,'surface_finish_ra':1.4,'assembly_clearance_mm':0.29,'overhang_angle_deg':34.0,'min_feature_size_mm':0.9,'aspect_ratio':2.9,'part_weight_kg':0.21,'vent_aspect_ratio':0.0,'cooling_channel_mm':0.0,'ip_groove_mm':1.6},
    {'product_type':'ADAS','label':'FAIL','wall_thickness_mm':1.3,'fillet_radius_mm':0.5,'draft_angle_deg':0.9,'hole_diameter_mm':3.5,'rib_height_mm':18.0,'rib_thickness_mm':2.2,'tolerance_mm':0.13,'material_density':1.09,'surface_finish_ra':3.0,'assembly_clearance_mm':0.08,'overhang_angle_deg':55.0,'min_feature_size_mm':0.35,'aspect_ratio':8.0,'part_weight_kg':0.32,'vent_aspect_ratio':0.0,'cooling_channel_mm':0.0,'ip_groove_mm':0.7},
    {'product_type':'ADAS','label':'PASS','wall_thickness_mm':2.8,'fillet_radius_mm':1.1,'draft_angle_deg':2.2,'hole_diameter_mm':8.0,'rib_height_mm':7.0,'rib_thickness_mm':3.3,'tolerance_mm':0.048,'material_density':1.05,'surface_finish_ra':1.2,'assembly_clearance_mm':0.34,'overhang_angle_deg':30.0,'min_feature_size_mm':1.2,'aspect_ratio':2.5,'part_weight_kg':0.19,'vent_aspect_ratio':0.0,'cooling_channel_mm':0.0,'ip_groove_mm':1.7},
    {'product_type':'ADAS','label':'FAIL','wall_thickness_mm':1.6,'fillet_radius_mm':0.55,'draft_angle_deg':1.2,'hole_diameter_mm':5.0,'rib_height_mm':14.0,'rib_thickness_mm':2.6,'tolerance_mm':0.09,'material_density':1.07,'surface_finish_ra':2.0,'assembly_clearance_mm':0.14,'overhang_angle_deg':50.0,'min_feature_size_mm':0.5,'aspect_ratio':6.0,'part_weight_kg':0.30,'vent_aspect_ratio':0.0,'cooling_channel_mm':0.0,'ip_groove_mm':0.85},
    {'product_type':'ADAS','label':'PASS','wall_thickness_mm':2.5,'fillet_radius_mm':0.98,'draft_angle_deg':2.0,'hole_diameter_mm':7.2,'rib_height_mm':7.3,'rib_thickness_mm':3.15,'tolerance_mm':0.053,'material_density':1.055,'surface_finish_ra':1.32,'assembly_clearance_mm':0.30,'overhang_angle_deg':32.0,'min_feature_size_mm':0.95,'aspect_ratio':2.75,'part_weight_kg':0.20,'vent_aspect_ratio':0.0,'cooling_channel_mm':0.0,'ip_groove_mm':1.55},
    {'product_type':'ADAS','label':'FAIL','wall_thickness_mm':1.4,'fillet_radius_mm':0.52,'draft_angle_deg':1.0,'hole_diameter_mm':4.0,'rib_height_mm':16.5,'rib_thickness_mm':2.3,'tolerance_mm':0.11,'material_density':1.08,'surface_finish_ra':2.6,'assembly_clearance_mm':0.10,'overhang_angle_deg':53.0,'min_feature_size_mm':0.4,'aspect_ratio':7.2,'part_weight_kg':0.29,'vent_aspect_ratio':0.0,'cooling_channel_mm':0.0,'ip_groove_mm':0.75},
    {'product_type':'ADAS','label':'PASS','wall_thickness_mm':2.7,'fillet_radius_mm':1.05,'draft_angle_deg':2.15,'hole_diameter_mm':7.8,'rib_height_mm':7.1,'rib_thickness_mm':3.25,'tolerance_mm':0.05,'material_density':1.048,'surface_finish_ra':1.25,'assembly_clearance_mm':0.32,'overhang_angle_deg':31.0,'min_feature_size_mm':1.1,'aspect_ratio':2.65,'part_weight_kg':0.20,'vent_aspect_ratio':0.0,'cooling_channel_mm':0.0,'ip_groove_mm':1.62},
    {'product_type':'ADAS','label':'FAIL','wall_thickness_mm':1.5,'fillet_radius_mm':0.58,'draft_angle_deg':1.15,'hole_diameter_mm':4.8,'rib_height_mm':15.5,'rib_thickness_mm':2.55,'tolerance_mm':0.095,'material_density':1.075,'surface_finish_ra':2.1,'assembly_clearance_mm':0.13,'overhang_angle_deg':51.0,'min_feature_size_mm':0.48,'aspect_ratio':6.2,'part_weight_kg':0.29,'vent_aspect_ratio':0.0,'cooling_channel_mm':0.0,'ip_groove_mm':0.82},
    {'product_type':'ADAS','label':'PASS','wall_thickness_mm':2.9,'fillet_radius_mm':1.15,'draft_angle_deg':2.3,'hole_diameter_mm':8.5,'rib_height_mm':6.8,'rib_thickness_mm':3.4,'tolerance_mm':0.045,'material_density':1.042,'surface_finish_ra':1.15,'assembly_clearance_mm':0.36,'overhang_angle_deg':29.0,'min_feature_size_mm':1.4,'aspect_ratio':2.3,'part_weight_kg':0.21,'vent_aspect_ratio':0.0,'cooling_channel_mm':0.0,'ip_groove_mm':1.75},
    # STRUCTURAL (12)
    {'product_type':'STRUCTURAL','label':'PASS','wall_thickness_mm':3.5,'fillet_radius_mm':2.0,'draft_angle_deg':3.0,'hole_diameter_mm':15.0,'rib_height_mm':12.0,'rib_thickness_mm':4.8,'tolerance_mm':0.08,'material_density':2.76,'surface_finish_ra':2.0,'assembly_clearance_mm':0.40,'overhang_angle_deg':40.0,'min_feature_size_mm':2.2,'aspect_ratio':4.0,'part_weight_kg':2.25,'vent_aspect_ratio':0.0,'cooling_channel_mm':0.0,'ip_groove_mm':0.0},
    {'product_type':'STRUCTURAL','label':'FAIL','wall_thickness_mm':1.0,'fillet_radius_mm':0.25,'draft_angle_deg':0.5,'hole_diameter_mm':2.5,'rib_height_mm':22.0,'rib_thickness_mm':2.0,'tolerance_mm':0.18,'material_density':2.80,'surface_finish_ra':4.0,'assembly_clearance_mm':0.06,'overhang_angle_deg':60.0,'min_feature_size_mm':0.3,'aspect_ratio':9.0,'part_weight_kg':1.20,'vent_aspect_ratio':0.0,'cooling_channel_mm':0.0,'ip_groove_mm':0.0},
    {'product_type':'STRUCTURAL','label':'PASS','wall_thickness_mm':3.2,'fillet_radius_mm':1.8,'draft_angle_deg':2.8,'hole_diameter_mm':14.0,'rib_height_mm':11.0,'rib_thickness_mm':4.5,'tolerance_mm':0.075,'material_density':2.72,'surface_finish_ra':1.8,'assembly_clearance_mm':0.38,'overhang_angle_deg':38.0,'min_feature_size_mm':2.0,'aspect_ratio':3.8,'part_weight_kg':2.10,'vent_aspect_ratio':0.0,'cooling_channel_mm':0.0,'ip_groove_mm':0.0},
    {'product_type':'STRUCTURAL','label':'PASS','wall_thickness_mm':2.8,'fillet_radius_mm':1.5,'draft_angle_deg':2.5,'hole_diameter_mm':12.0,'rib_height_mm':10.5,'rib_thickness_mm':4.2,'tolerance_mm':0.07,'material_density':2.68,'surface_finish_ra':1.9,'assembly_clearance_mm':0.35,'overhang_angle_deg':37.0,'min_feature_size_mm':1.8,'aspect_ratio':3.5,'part_weight_kg':1.95,'vent_aspect_ratio':0.0,'cooling_channel_mm':0.0,'ip_groove_mm':0.0},
    {'product_type':'STRUCTURAL','label':'FAIL','wall_thickness_mm':1.1,'fillet_radius_mm':0.3,'draft_angle_deg':0.6,'hole_diameter_mm':3.0,'rib_height_mm':21.0,'rib_thickness_mm':2.1,'tolerance_mm':0.17,'material_density':2.78,'surface_finish_ra':3.8,'assembly_clearance_mm':0.07,'overhang_angle_deg':59.0,'min_feature_size_mm':0.32,'aspect_ratio':8.8,'part_weight_kg':1.25,'vent_aspect_ratio':0.0,'cooling_channel_mm':0.0,'ip_groove_mm':0.0},
    {'product_type':'STRUCTURAL','label':'PASS','wall_thickness_mm':3.4,'fillet_radius_mm':1.9,'draft_angle_deg':2.9,'hole_diameter_mm':14.5,'rib_height_mm':11.5,'rib_thickness_mm':4.6,'tolerance_mm':0.078,'material_density':2.75,'surface_finish_ra':1.95,'assembly_clearance_mm':0.39,'overhang_angle_deg':39.0,'min_feature_size_mm':2.1,'aspect_ratio':3.9,'part_weight_kg':2.20,'vent_aspect_ratio':0.0,'cooling_channel_mm':0.0,'ip_groove_mm':0.0},
    {'product_type':'STRUCTURAL','label':'FAIL','wall_thickness_mm':1.2,'fillet_radius_mm':0.35,'draft_angle_deg':0.65,'hole_diameter_mm':3.2,'rib_height_mm':20.5,'rib_thickness_mm':2.15,'tolerance_mm':0.165,'material_density':2.79,'surface_finish_ra':3.5,'assembly_clearance_mm':0.075,'overhang_angle_deg':58.0,'min_feature_size_mm':0.35,'aspect_ratio':8.5,'part_weight_kg':1.28,'vent_aspect_ratio':0.0,'cooling_channel_mm':0.0,'ip_groove_mm':0.0},
    {'product_type':'STRUCTURAL','label':'PASS','wall_thickness_mm':3.0,'fillet_radius_mm':1.6,'draft_angle_deg':2.6,'hole_diameter_mm':13.0,'rib_height_mm':10.8,'rib_thickness_mm':4.3,'tolerance_mm':0.072,'material_density':2.70,'surface_finish_ra':1.85,'assembly_clearance_mm':0.36,'overhang_angle_deg':38.5,'min_feature_size_mm':1.9,'aspect_ratio':3.6,'part_weight_kg':2.00,'vent_aspect_ratio':0.0,'cooling_channel_mm':0.0,'ip_groove_mm':0.0},
    {'product_type':'STRUCTURAL','label':'PASS','wall_thickness_mm':2.9,'fillet_radius_mm':1.55,'draft_angle_deg':2.55,'hole_diameter_mm':12.5,'rib_height_mm':10.6,'rib_thickness_mm':4.25,'tolerance_mm':0.071,'material_density':2.69,'surface_finish_ra':1.88,'assembly_clearance_mm':0.355,'overhang_angle_deg':38.2,'min_feature_size_mm':1.85,'aspect_ratio':3.55,'part_weight_kg':1.98,'vent_aspect_ratio':0.0,'cooling_channel_mm':0.0,'ip_groove_mm':0.0},
    {'product_type':'STRUCTURAL','label':'FAIL','wall_thickness_mm':1.15,'fillet_radius_mm':0.32,'draft_angle_deg':0.62,'hole_diameter_mm':3.1,'rib_height_mm':20.8,'rib_thickness_mm':2.12,'tolerance_mm':0.168,'material_density':2.785,'surface_finish_ra':3.65,'assembly_clearance_mm':0.072,'overhang_angle_deg':58.5,'min_feature_size_mm':0.33,'aspect_ratio':8.65,'part_weight_kg':1.26,'vent_aspect_ratio':0.0,'cooling_channel_mm':0.0,'ip_groove_mm':0.0},
    {'product_type':'STRUCTURAL','label':'PASS','wall_thickness_mm':3.3,'fillet_radius_mm':1.85,'draft_angle_deg':2.85,'hole_diameter_mm':14.2,'rib_height_mm':11.2,'rib_thickness_mm':4.55,'tolerance_mm':0.077,'material_density':2.74,'surface_finish_ra':1.92,'assembly_clearance_mm':0.385,'overhang_angle_deg':39.5,'min_feature_size_mm':2.05,'aspect_ratio':3.85,'part_weight_kg':2.15,'vent_aspect_ratio':0.0,'cooling_channel_mm':0.0,'ip_groove_mm':0.0},
    {'product_type':'STRUCTURAL','label':'PASS','wall_thickness_mm':2.75,'fillet_radius_mm':1.45,'draft_angle_deg':2.45,'hole_diameter_mm':11.5,'rib_height_mm':10.2,'rib_thickness_mm':4.1,'tolerance_mm':0.068,'material_density':2.66,'surface_finish_ra':1.82,'assembly_clearance_mm':0.345,'overhang_angle_deg':37.5,'min_feature_size_mm':1.75,'aspect_ratio':3.45,'part_weight_kg':1.92,'vent_aspect_ratio':0.0,'cooling_channel_mm':0.0,'ip_groove_mm':0.0},
]

# ── Design rules (12 Varroc rules) ───────────────────────────────────────────
DESIGN_RULES = {
    "R01": {"name":"Minimum Wall Thickness","context":"Headlamp housing / Motor casing","parameter":"wall_thickness_mm","condition":lambda v:v>=2.0,"severity":"CRITICAL","message":"Wall {val:.2f}mm < 2.0mm — thermal/vibration failure risk.","standard":"DFM-VAR-001","applies_to":["LIGHTING","EV","ADAS","STRUCTURAL"]},
    "R02": {"name":"Thermal Vent Aspect Ratio","context":"Headlamp condensate vents","parameter":"vent_aspect_ratio","condition":lambda v:(v<=4.0) if v>0 else True,"severity":"CRITICAL","message":"Vent AR {val:.1f}:1 > 4:1 — moisture trap (Ref: PS5 condensate).","standard":"LGT-VAR-001","applies_to":["LIGHTING"]},
    "R03": {"name":"Lens-Housing Assembly Clearance","context":"Headlamp / DRL assembly stack","parameter":"assembly_clearance_mm","condition":lambda v:0.20<=v<=0.50,"severity":"CRITICAL","message":"Clearance {val:.2f}mm not in 0.20–0.50mm — lens fracture or IP breach.","standard":"ASM-VAR-001","applies_to":["LIGHTING"]},
    "R04": {"name":"Draft Angle","context":"Injection-moulded Varroc parts","parameter":"draft_angle_deg","condition":lambda v:v>=1.5,"severity":"MAJOR","message":"Draft {val:.1f}° < 1.5° — mold release failure in high-volume production.","standard":"DFM-VAR-002","applies_to":["LIGHTING","EV","ADAS","STRUCTURAL"]},
    "R05": {"name":"Fillet Radius","context":"Motor housing / structural stress zones","parameter":"fillet_radius_mm","condition":lambda v:v>=0.8,"severity":"MAJOR","message":"Fillet {val:.2f}mm < 0.8mm — fatigue crack initiation at stress zones.","standard":"DFM-VAR-003","applies_to":["EV","STRUCTURAL"]},
    "R06": {"name":"Cooling Channel Width","context":"PMSM motor cooling jacket","parameter":"cooling_channel_mm","condition":lambda v:(v>=3.0) if v>0 else True,"severity":"CRITICAL","message":"Cooling ch {val:.1f}mm < 3.0mm — demagnetisation risk (Ref: PS4 PMSM).","standard":"EV-VAR-001","applies_to":["EV"]},
    "R07": {"name":"Overhang Angle","context":"Bracket / body structural / die-cast parts","parameter":"overhang_angle_deg","condition":lambda v:v<=45.0,"severity":"MAJOR","message":"Overhang {val:.1f}° > 45° — support structures required, adds cost.","standard":"STR-VAR-001","applies_to":["LIGHTING","EV","ADAS","STRUCTURAL"]},
    "R08": {"name":"Minimum Feature Size","context":"EV connector / PCB brackets / ADAS mounts","parameter":"min_feature_size_mm","condition":lambda v:v>=0.6,"severity":"MAJOR","message":"Feature {val:.2f}mm < 0.6mm — beyond Varroc tooling capability.","standard":"MFG-VAR-001","applies_to":["EV","ADAS","STRUCTURAL"]},
    "R09": {"name":"Surface Finish — Sealing Surfaces","context":"Headlamp / ECU sealing and mating surfaces","parameter":"surface_finish_ra","condition":lambda v:v<=1.6,"severity":"MINOR","message":"Ra {val:.1f}μm > 1.6μm — IP67 seal quality insufficient.","standard":"SFC-VAR-001","applies_to":["LIGHTING","ADAS"]},
    "R10": {"name":"Rib Height-to-Thickness Ratio","context":"All injection-moulded Varroc parts","parameter":None,"condition":None,"severity":"MAJOR","message":"Rib H/T {val:.1f}:1 > 3:1 — sink marks on class-A surfaces.","standard":"DFM-VAR-004","applies_to":["LIGHTING","EV","ADAS","STRUCTURAL"],"custom":True},
    "R11": {"name":"Part Aspect Ratio","context":"Structural brackets / motor mounts","parameter":"aspect_ratio","condition":lambda v:v<=5.0,"severity":"MAJOR","message":"AR {val:.1f}:1 > 5:1 — NVH buckling risk (AIS-056).","standard":"STR-VAR-002","applies_to":["STRUCTURAL","EV"]},
    "R12": {"name":"IP Rating Sealing Groove","context":"Sensor / ECU / connector enclosures","parameter":"ip_groove_mm","condition":lambda v:(v>=1.2) if v>0 else True,"severity":"MAJOR","message":"IP groove {val:.1f}mm < 1.2mm — O-ring won't fit, IP67 fails.","standard":"IP-VAR-001","applies_to":["ADAS","EV"]},
}


def _train_model():
    """Train GBM on embedded dataset."""
    import pandas as pd
    df = pd.DataFrame(TRAINING_DATA)
    df['label_enc'] = (df['label'] == 'PASS').astype(int)
    X = df[FEATURE_COLUMNS].values
    y = df['label_enc'].values
    from sklearn.model_selection import train_test_split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = Pipeline([
        ('sc', StandardScaler()),
        ('clf', GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def _get_model():
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'smartcad_gbm.pkl')
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    model = _train_model()
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    return model


# Load model once at import time
_GBM = _get_model()


def run_rule_engine(design: dict, product_type: str = None) -> dict:
    violations, passed, skipped = [], [], []
    for rid, rule in DESIGN_RULES.items():
        if product_type and product_type not in rule.get("applies_to", []):
            skipped.append(rid)
            continue
        if rule.get("custom"):
            if "rib_height_mm" in design and "rib_thickness_mm" in design:
                ratio = design["rib_height_mm"] / max(design["rib_thickness_mm"], 0.01)
                entry = dict(rule_id=rid, rule_name=rule["name"], context=rule["context"],
                             severity=rule["severity"], message=rule["message"].format(val=ratio),
                             standard=rule["standard"], measured_value=round(ratio, 2))
                if ratio > 3.0:
                    violations.append(entry)
                else:
                    passed.append(dict(rule_id=rid, rule_name=rule["name"]))
            continue
        val = design.get(rule["parameter"])
        if val is None:
            skipped.append(rid)
            continue
        entry = dict(rule_id=rid, rule_name=rule["name"], context=rule["context"],
                     severity=rule["severity"], message=rule["message"].format(val=val),
                     standard=rule["standard"], measured_value=val)
        if not rule["condition"](val):
            violations.append(entry)
        else:
            passed.append(dict(rule_id=rid, rule_name=rule["name"]))

    n_crit = sum(1 for v in violations if v["severity"] == "CRITICAL")
    n_maj  = sum(1 for v in violations if v["severity"] == "MAJOR")
    n_min  = sum(1 for v in violations if v["severity"] == "MINOR")

    if n_crit > 0:       verdict, risk = "FAIL", "HIGH"
    elif n_maj >= 2:     verdict, risk = "FAIL", "MEDIUM-HIGH"
    elif n_maj == 1:     verdict, risk = "WARNING", "MEDIUM"
    else:                verdict, risk = "PASS", "LOW"

    return dict(violations=violations, passed_rules=passed, skipped_rules=skipped,
                rule_verdict=verdict, risk_level=risk,
                critical_count=n_crit, major_count=n_maj, minor_count=n_min)


def ml_predict(design: dict) -> dict:
    fv   = np.array([[design.get(c, 0.0) for c in FEATURE_COLUMNS]])
    pred = _GBM.predict(fv)[0]
    prob = _GBM.predict_proba(fv)[0]
    return {
        'ml_verdict':  'PASS' if pred == 1 else 'FAIL',
        'pass_prob':   round(float(prob[1]) * 100, 1),
        'fail_prob':   round(float(prob[0]) * 100, 1),
        'confidence':  round(float(max(prob)) * 100, 1),
    }


def fuse_verdicts(rule_result: dict, ml_result: dict) -> dict:
    rv, mv, mc, cr = (rule_result['rule_verdict'], ml_result['ml_verdict'],
                      ml_result['confidence'], rule_result['critical_count'])
    if cr > 0:
        return dict(verdict='FAIL', method='Rule Override — Critical Varroc Safety Violation', confidence=99.0)
    if rv == 'PASS' and mv == 'PASS':
        return dict(verdict='PASS', method='Consensus — Rule Engine + GBM agree PASS', confidence=round(min(99, (mc+92)/2), 1))
    if rv in ('FAIL','WARNING') and mv == 'FAIL':
        return dict(verdict='FAIL', method='Consensus — Rule Engine + GBM agree FAIL', confidence=round(min(99, (mc+88)/2), 1))
    if rv == 'PASS' and mv == 'FAIL':
        if mc > 75:
            return dict(verdict='FAIL', method='GBM Override — Interaction pattern beyond explicit rules', confidence=mc)
        return dict(verdict='WARNING', method='Conflict — Escalate to Varroc Design Lead', confidence=mc)
    if rv in ('FAIL','WARNING') and mv == 'PASS':
        return dict(verdict='WARNING', method='Partial Flag — Rule concern, GBM uncertain; manual review', confidence=mc)
    return dict(verdict='WARNING', method='Ambiguous — Review Recommended', confidence=60.0)


def validate_design(design: dict, product_type: str = None) -> dict:
    rule_r = run_rule_engine(design, product_type=product_type)
    ml_r   = ml_predict(design)
    fusion = fuse_verdicts(rule_r, ml_r)
    return dict(
        product_type=product_type or 'GENERAL',
        rule_result=rule_r,
        ml_result=ml_r,
        fusion=fusion,
    )
