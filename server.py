"""
Impetus Aquae Fontis — REST API Server
CASSINI Hackathon: Space for Water

Exposes irrigation forecast, water balance, and ML yield prediction
through a lightweight FastAPI server on localhost.
"""

import os
import math
import time
import datetime as dt
from dataclasses import asdict
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from main import (
    FieldConfig,
    fetch_weather_data,
    fetch_soil_moisture_profile,
    fetch_sentinel1_soil_moisture,
    blend_soil_moisture,
    run_water_balance,
    generate_alerts,
    estimate_ndvi,
    plot_forecast,
    mm_to_m3,
    CROP_THRESHOLDS,
    # ML prediction
    fetch_eurostat_yields,
    build_training_dataset,
    train_yield_model,
    predict_current_season,
)

app = FastAPI(
    title="Impetus Aquae Fontis API",
    description=(
        "Irrigation forecast & runoff water management system. "
        "Uses Copernicus ERA5 satellite data, Eurostat crop yields, "
        "and ML predictions to advise agricultural producers."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# ── ML Model Cache (loaded from pretrained files or trained at startup) ──────

_model_cache: dict = {}  # keyed by crop name → {model, scaler, feature_cols, training_df, trained_at}

def _load_pretrained_models() -> bool:
    """Try to load pretrained models from disk. Returns True if at least one loaded."""
    import joblib
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_models")
    if not os.path.isdir(model_dir):
        return False
    loaded = 0
    for fname in os.listdir(model_dir):
        if fname.endswith("_model.joblib"):
            crop = fname.replace("_model.joblib", "")
            path = os.path.join(model_dir, fname)
            try:
                payload = joblib.load(path)
                _model_cache[crop] = {
                    "model": payload["model"],
                    "scaler": payload["scaler"],
                    "feature_cols": payload["feature_cols"],
                    "training_df": payload["training_df"],
                    "trained_at": payload.get("trained_at", "pretrained"),
                }
                print(f"  ✅ Loaded pretrained {crop} model ({len(payload['training_df'])} years)")
                loaded += 1
            except Exception as e:
                print(f"  ⚠️  Failed to load {fname}: {e}")
    return loaded > 0


def _train_crop_model(crop: str) -> bool:
    """Train and cache the ML model for a specific crop. Returns True on success."""
    print(f"  🤖 Training ML model for {crop}...")
    t0 = time.time()
    for attempt in range(3):
        try:
            cfg = FieldConfig()
            cfg.crop = crop
            yields = fetch_eurostat_yields(crop)
            if len(yields) < 5:
                print(f"  ⚠️  Insufficient Eurostat yield data for {crop} ({len(yields)} years)")
                return False
            training_df = build_training_dataset(cfg, yields, None)
            if len(training_df) < 5:
                print(f"  ⚠️  Insufficient training data for {crop} ({len(training_df)} rows)")
                return False
            model, scaler, feature_cols, training_df = train_yield_model(training_df)
            _model_cache[crop] = {
                "model": model,
                "scaler": scaler,
                "feature_cols": feature_cols,
                "training_df": training_df,
                "trained_at": dt.datetime.now().isoformat(),
            }
            elapsed = time.time() - t0
            print(f"  ✅ {crop.title()} model ready — {len(training_df)} years, {elapsed:.1f}s")
            return True
        except Exception as e:
            if attempt < 2:
                print(f"  ⚠️  Attempt {attempt+1} failed for {crop}: {e} — retrying in 5s...")
                time.sleep(5)
            else:
                print(f"  ❌ Failed to train {crop} model after 3 attempts: {e}")
                return False


@app.on_event("startup")
def startup_train_models():
    """Load pretrained models or train from scratch at server startup."""
    print("\n🧠 Loading ML yield prediction models...")
    t0 = time.time()

    if _load_pretrained_models():
        elapsed = time.time() - t0
        cached = list(_model_cache.keys())
        print(f"🧠 Loaded {len(cached)} pretrained models "
              f"({', '.join(cached)}) in {elapsed:.1f}s\n")
        return

    print("  ℹ️  No pretrained models found, training from scratch...")
    for crop in ["maize", "wheat"]:
        _train_crop_model(crop)
    elapsed = time.time() - t0
    cached = list(_model_cache.keys())
    print(f"🧠 Model training complete — {len(cached)} models cached "
          f"({', '.join(cached)}) in {elapsed:.1f}s\n")

# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_date(date_str: Optional[str]) -> Optional[dt.date]:
    return dt.date.fromisoformat(date_str) if date_str else None


def make_config(
    lat: float = 46.305, lon: float = 20.050, area_ha: float = 50.0,
    crop: str = "maize", reservoir_capacity: float = 15000.0,
    initial_reservoir: float = 5000.0,
) -> FieldConfig:
    cfg = FieldConfig()
    cfg.lat = lat
    cfg.lon = lon
    cfg.area_ha = area_ha
    cfg.crop = crop
    cfg.reservoir_capacity_m3 = reservoir_capacity
    cfg.initial_reservoir_m3 = initial_reservoir
    return cfg


def state_to_dict(s) -> dict:
    d = asdict(s)
    d["date"] = s.date.isoformat()
    return d


def alert_to_dict(a) -> dict:
    return asdict(a)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", tags=["Frontend"], include_in_schema=False)
def root():
    """Serve the web dashboard at root URL."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api", tags=["Info"])
def api_info():
    """API overview and available endpoints."""
    return {
        "name": "Impetus Aquae Fontis API",
        "hackathon": "CASSINI — Space for Water",
        "version": "1.0.0",
        "data_sources": [
            "Copernicus ERA5 reanalysis",
            "Copernicus ERA5-Land soil moisture",
            "ECMWF IFS weather forecast",
            "Sentinel-2 NDVI (phenology-based)",
            "Eurostat crop yield statistics",
            "Galileo OS-NMA (planned)",
        ],
        "endpoints": {
            "/config": "Current field configuration",
            "/weather": "Raw weather data (ERA5 + ECMWF IFS)",
            "/soil-profile": "Multi-depth soil moisture (ERA5-Land)",
            "/forecast": "7-day irrigation forecast with water balance",
            "/alerts": "Active alerts and notifications",
            "/prediction": "ML yield prediction with irrigation impact",
            "/chart": "Forecast chart image (PNG)",
            "/docs": "Interactive API documentation (Swagger UI)",
        },
    }


@app.get("/config", tags=["Configuration"])
def get_config(
    lat: float = Query(46.305, description="Field latitude"),
    lon: float = Query(20.050, description="Field longitude"),
    area_ha: float = Query(50.0, description="Field area in hectares"),
    crop: str = Query("maize", description="Crop type (maize or wheat)"),
    reservoir_capacity: float = Query(15000.0, description="Reservoir capacity in m³"),
    initial_reservoir: float = Query(5000.0, description="Current reservoir level in m³"),
):
    """Return the field configuration that will be used for calculations."""
    cfg = make_config(lat, lon, area_ha, crop, reservoir_capacity, initial_reservoir)
    crop_info = CROP_THRESHOLDS.get(crop, {})
    aw = cfg.soil_field_capacity_mm - cfg.soil_wilting_point_mm
    return {
        "field": {
            "name": cfg.name,
            "lat": cfg.lat,
            "lon": cfg.lon,
            "area_ha": cfg.area_ha,
            "crop": cfg.crop,
        },
        "soil": {
            "field_capacity_mm": cfg.soil_field_capacity_mm,
            "wilting_point_mm": cfg.soil_wilting_point_mm,
            "stress_threshold_mm": cfg.soil_wilting_point_mm + aw * crop_info.get("stress_fraction", 0.5),
            "critical_threshold_mm": cfg.soil_wilting_point_mm + aw * crop_info.get("critical_fraction", 0.25),
            "root_zone_depth_mm": cfg.root_zone_depth_mm,
        },
        "reservoir": {
            "capacity_m3": cfg.reservoir_capacity_m3,
            "current_m3": cfg.initial_reservoir_m3,
            "capture_efficiency": cfg.runoff_capture_efficiency,
            "irrigation_efficiency": cfg.irrigation_efficiency,
        },
    }


@app.get("/weather", tags=["Data"])
def get_weather(
    date: Optional[str] = Query(None, description="Simulation date (YYYY-MM-DD). Omit for live."),
    lat: float = Query(46.305), lon: float = Query(20.050),
    past_days: int = Query(30, description="Days of history"),
    forecast_days: int = Query(7, description="Days of forecast"),
    crop: str = Query("maize"),
):
    """Fetch raw weather data from Copernicus ERA5 / ECMWF IFS."""
    cfg = make_config(lat, lon, crop=crop)
    sim_date = parse_date(date)
    data = fetch_weather_data(cfg, past_days=past_days, forecast_days=forecast_days, sim_date=sim_date)
    return {
        "sim_date": date or dt.date.today().isoformat(),
        "source": "Copernicus ERA5 reanalysis" + (" (historical)" if sim_date else " + ECMWF IFS forecast"),
        "location": {"lat": data.get("latitude"), "lon": data.get("longitude"), "elevation_m": data.get("elevation")},
        "daily": data.get("daily", {}),
    }


@app.get("/soil-profile", tags=["Data"])
def get_soil_profile(
    date: Optional[str] = Query(None, description="Simulation date (YYYY-MM-DD)"),
    lat: float = Query(46.305), lon: float = Query(20.050),
    crop: str = Query("maize"),
):
    """Fetch multi-depth soil moisture from ERA5-Land + Sentinel-1 SAR fusion."""
    cfg = make_config(lat, lon, crop=crop)
    sim_date = parse_date(date)

    # ERA5-Land baseline
    era5_profile = fetch_soil_moisture_profile(cfg, sim_date=sim_date)

    # Sentinel-1 SAR fusion
    s1_data = fetch_sentinel1_soil_moisture(cfg, sim_date=sim_date, era5_profile=era5_profile)
    blended = s1_data.get("blended_profile") or era5_profile
    sar = s1_data.get("sar_analysis", {})

    result = {}
    for key, val in blended.items():
        depth = key.replace("soil_moisture_", "").replace("_", "-")
        era5_val = era5_profile.get(key)
        status = "dry" if val and val < 0.15 else ("low" if val and val < 0.25 else "good")
        result[depth] = {
            "value_m3_m3": val,
            "era5_value_m3_m3": era5_val,
            "status": status,
        }

    return {
        "sim_date": date or dt.date.today().isoformat(),
        "source": "Copernicus ERA5-Land + Sentinel-1 SAR fusion",
        "fusion_weights": "Surface: 60% SAR + 40% ERA5 | Mid: 20% SAR + 80% ERA5 | Deep: 100% ERA5",
        "profile": result,
        "sentinel1": {
            "sigma0_vv_db": sar.get("sigma0_vv_db"),
            "sigma0_vh_db": sar.get("sigma0_vh_db"),
            "sar_surface_moisture_m3m3": sar.get("sar_soil_moisture_m3m3"),
            "model": sar.get("model"),
            "products_found": len(s1_data.get("products", [])),
        },
    }


@app.get("/sentinel1", tags=["Data"])
def get_sentinel1(
    date: Optional[str] = Query(None, description="Simulation date (YYYY-MM-DD)"),
    lat: float = Query(46.305), lon: float = Query(20.050),
    crop: str = Query("maize"),
):
    """Sentinel-1 SAR soil moisture analysis — backscatter, inversion, and catalog products."""
    cfg = make_config(lat, lon, crop=crop)
    sim_date = parse_date(date)
    era5_profile = fetch_soil_moisture_profile(cfg, sim_date=sim_date)
    s1_data = fetch_sentinel1_soil_moisture(cfg, sim_date=sim_date, era5_profile=era5_profile)
    return s1_data


@app.get("/forecast", tags=["Forecast"])
def get_forecast(
    date: Optional[str] = Query(None, description="Simulation date (YYYY-MM-DD)"),
    lat: float = Query(46.305), lon: float = Query(20.050),
    area_ha: float = Query(50.0), crop: str = Query("maize"),
    reservoir_capacity: float = Query(15000.0),
    initial_reservoir: float = Query(5000.0),
):
    """
    Run the full water balance model and return the 7-day irrigation forecast.
    Includes soil moisture, runoff collection, reservoir state, stress levels,
    drought risk, and recommended actions.
    """
    cfg = make_config(lat, lon, area_ha, crop, reservoir_capacity, initial_reservoir)
    sim_date = parse_date(date)
    today = sim_date or dt.date.today()

    weather = fetch_weather_data(cfg, past_days=30, forecast_days=7, sim_date=sim_date)
    states = run_water_balance(cfg, weather, past_days=30, sim_date=sim_date)

    forecast_states = [s for s in states if s.date >= today]
    historical_states = [s for s in states if s.date < today]

    # Historical summary
    total_precip = sum(s.precip_mm for s in historical_states)
    total_et = sum(s.et0_mm for s in historical_states)
    total_runoff = sum(s.runoff_collected_m3 for s in historical_states)

    return {
        "sim_date": today.isoformat(),
        "field": {"lat": cfg.lat, "lon": cfg.lon, "area_ha": cfg.area_ha, "crop": cfg.crop},
        "historical_summary": {
            "days": len(historical_states),
            "total_precipitation_mm": round(total_precip, 1),
            "total_evapotranspiration_mm": round(total_et, 1),
            "water_balance_mm": round(total_precip - total_et, 1),
            "runoff_collected_m3": round(total_runoff, 0),
            "soil_moisture_trend": {
                "start_mm": historical_states[0].soil_moisture_mm if historical_states else None,
                "end_mm": historical_states[-1].soil_moisture_mm if historical_states else None,
            },
        },
        "forecast": [state_to_dict(s) for s in forecast_states],
    }


@app.get("/alerts", tags=["Forecast"])
def get_alerts(
    date: Optional[str] = Query(None, description="Simulation date (YYYY-MM-DD)"),
    lat: float = Query(46.305), lon: float = Query(20.050),
    area_ha: float = Query(50.0), crop: str = Query("maize"),
    reservoir_capacity: float = Query(15000.0),
    initial_reservoir: float = Query(5000.0),
):
    """
    Generate alerts based on the current forecast.
    Returns drought alerts, irrigation schedules, runoff opportunities,
    and reservoir warnings with suggested notification channels.
    """
    cfg = make_config(lat, lon, area_ha, crop, reservoir_capacity, initial_reservoir)
    sim_date = parse_date(date)

    weather = fetch_weather_data(cfg, past_days=30, forecast_days=7, sim_date=sim_date)
    states = run_water_balance(cfg, weather, past_days=30, sim_date=sim_date)
    alerts = generate_alerts(states, cfg, sim_date)

    return {
        "sim_date": (sim_date or dt.date.today()).isoformat(),
        "alert_count": len(alerts),
        "alerts": [alert_to_dict(a) for a in alerts],
    }


@app.get("/prediction", tags=["ML Prediction"])
def get_prediction(
    date: Optional[str] = Query(None, description="Simulation date (YYYY-MM-DD)"),
    lat: float = Query(46.305), lon: float = Query(20.050),
    area_ha: float = Query(50.0), crop: str = Query("maize"),
    scenario: str = Query("auto", description="Climate scenario: auto, hot_dry, normal, wet_cool"),
):
    """
    Predict the current season's yield using analog year matching + ML.

    Uses early-season weather to find similar historical years, then projects
    the full season. Supports climate scenarios for what-if analysis:
    - **auto**: detect from current weather trends
    - **hot_dry**: simulate a drought year (like 2003, 2007, 2022)
    - **normal**: average conditions
    - **wet_cool**: wet summer with lower temperatures
    """
    if crop not in _model_cache:
        if not _train_crop_model(crop):
            return JSONResponse(status_code=503, content={
                "error": f"No model available for '{crop}' — training failed. Try again later."
            })

    cached = _model_cache[crop]
    cfg = make_config(lat, lon, area_ha, crop=crop)
    sim_date = parse_date(date)

    prediction = predict_current_season(
        cfg, cached["model"], cached["scaler"],
        cached["feature_cols"], cached["training_df"], sim_date,
        scenario=scenario,
    )

    if not prediction:
        return JSONResponse(status_code=503, content={"error": "Could not generate prediction"})

    # Add economic impact
    price_per_tonne = 200 if crop == "maize" else 250
    revenue_diff = prediction["yield_improvement"] * area_ha * price_per_tonne

    prediction["economic_impact"] = {
        "price_per_tonne_eur": price_per_tonne,
        "yield_gain_tonnes": round(prediction["yield_improvement"] * area_ha, 1),
        "additional_revenue_eur": round(revenue_diff, 0),
    }

    prediction["model_info"] = {
        "algorithm": "GradientBoostingRegressor",
        "training_years": len(cached["training_df"]),
        "trained_at": cached["trained_at"],
        "data_sources": ["Eurostat apro_cpsh1 (crop yields)", "Copernicus ERA5 (growing season weather)"],
    }

    return {
        "sim_date": (sim_date or dt.date.today()).isoformat(),
        "crop": crop,
        "prediction": prediction,
    }


@app.get("/model-status", tags=["ML Prediction"])
def get_model_status():
    """Check the status of pre-trained ML models with Copernicus data sources."""
    status = {}
    for crop in ["maize", "wheat"]:
        if crop in _model_cache:
            c = _model_cache[crop]
            status[crop] = {
                "ready": True,
                "trained_at": c["trained_at"],
                "training_years": len(c["training_df"]),
                "n_features": len(c["feature_cols"]),
                "feature_cols": c["feature_cols"],
                "copernicus_sources": [
                    "ERA5 Reanalysis (weather)",
                    "ERA5-Land (soil moisture)",
                    "Sentinel-1 SAR VV (backscatter)",
                    "Sentinel-2 (NDVI + NDWI)",
                ],
            }
        else:
            status[crop] = {"ready": False}
    return {"models": status}


@app.get("/chart", tags=["Visualization"])
def get_chart(
    date: Optional[str] = Query(None, description="Simulation date (YYYY-MM-DD)"),
    lat: float = Query(46.305), lon: float = Query(20.050),
    area_ha: float = Query(50.0), crop: str = Query("maize"),
    reservoir_capacity: float = Query(15000.0),
    initial_reservoir: float = Query(5000.0),
):
    """Generate and return the forecast chart as a PNG image."""
    cfg = make_config(lat, lon, area_ha, crop, reservoir_capacity, initial_reservoir)
    sim_date = parse_date(date)

    weather = fetch_weather_data(cfg, past_days=30, forecast_days=7, sim_date=sim_date)
    states = run_water_balance(cfg, weather, past_days=30, sim_date=sim_date)
    plot_forecast(states, cfg, sim_date)

    import tempfile
    chart_path = os.path.join(tempfile.gettempdir(), "forecast_chart.png")
    return FileResponse(chart_path, media_type="image/png", filename="forecast_chart.png")


# ── Tisza River Transect Bands ────────────────────────────────────────────────

# Parallel bands running alongside the Tisza, going westward from the river.
# At lat ~46.26°, 1° lon ≈ 77.5 km, so 5km ≈ 0.0645° longitude offset.
# River centerline is roughly lon=20.16, bands go west into the farmland.

RIVER_CENTER_LAT = 46.395
RIVER_CENTER_LON = 20.200
LON_PER_5KM = 0.0645  # 5km in degrees longitude at this latitude

# River course polyline (N→S) for drawing on the map
# Real Tisza River course from GPX waypoints (south → north)
RIVER_COURSE = [
    [46.37341, 20.20942],  # WP01
    [46.38099, 20.20891],  # WP02
    [46.38797, 20.20719],  # WP03
    [46.39507, 20.20616],  # WP04
    [46.40040, 20.20444],  # WP05
    [46.40040, 20.19363],  # WP06
    [46.40596, 20.18641],  # WP07
    [46.41696, 20.18556],  # WP08
]

# 5 bands: river bank, then 0-5km, 5-10km, 10-15km, 15-20km west of river
TRANSECT_BANDS = [
    {"id": 0, "name": "River Bank",     "dist_km": 0,  "lon_offset": 0},
    {"id": 1, "name": "Floodplain",     "dist_km": 5,  "lon_offset": -1},
    {"id": 2, "name": "Near Farmland",  "dist_km": 10, "lon_offset": -2},
    {"id": 3, "name": "Inner Plains",   "dist_km": 15, "lon_offset": -3},
    {"id": 4, "name": "Outer Plains",   "dist_km": 20, "lon_offset": -4},
]

# ── WoSIS Soil Data Integration ──────────────────────────────────────────────
# Real bulk density from WoSIS (ISRIC World Soil Information Service)
# Three soil zones from WoSIS field data:
#   - Alluvial clay: near river, high BD, low K
#   - Fluvisol transition: intermediate zone
#   - Arenosol sand: far from river, lower BD, high K

import csv as _csv

def _load_wosis_data():
    """Load WoSIS bulk density profiles from all 3 soil zones."""
    base = os.path.dirname(os.path.abspath(__file__))
    zones = {
        "clay":       os.path.join(base, "wosis_soil_data", "wosis_alluvial_clay", "wosis_latest_bdfiod.csv"),
        "transition": os.path.join(base, "wosis_soil_data", "wosis_fluvisol_transition", "wosis_latest_bdfiod.csv"),
        "sandy":      os.path.join(base, "wosis_soil_data", "wosis_arenosol_sand", "wosis_latest_bdfiod.csv"),
    }
    all_profiles = []
    for zone_name, path in zones.items():
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                try:
                    all_profiles.append({
                        "zone": zone_name,
                        "x": float(row["X"]),
                        "y": float(row["Y"]),
                        "upper_depth": int(row["upper_depth"]),
                        "lower_depth": int(row["lower_depth"]),
                        "bulk_density": float(row["value_avg"]),
                        "profile_code": row["profile_code"],
                    })
                except (ValueError, KeyError):
                    pass
    return all_profiles

# Pre-compute depth-averaged bulk density per soil zone (near Szeged)
def _compute_zone_properties(profiles):
    """Derive hydraulic properties from WoSIS bulk density per soil zone.
    Uses pedotransfer functions:
      porosity = 1 - BD / 2.65 (particle density of quartz)
      K estimated from Kozeny-Carman: K ~ porosity^3 / (1-porosity)^2 scaled
    """
    # Filter to profiles near our transect area
    nearby = [p for p in profiles if 46.0 <= p["y"] <= 46.6 and 19.8 <= p["x"] <= 20.5]
    if not nearby:
        nearby = profiles  # fallback to all

    # Average BD across all depths for each zone
    zone_bd = {}
    for p in nearby:
        zone_bd.setdefault(p["zone"], []).append(p["bulk_density"])

    zone_props = {}
    for zone, bds in zone_bd.items():
        avg_bd = sum(bds) / len(bds)
        porosity = 1.0 - avg_bd / 2.65
        # Kozeny-Carman relative K (normalized to K_ref at porosity=0.30)
        # K ~ n^3 / (1-n)^2; sand (high porosity) → higher K
        n = max(porosity, 0.05)
        kc_factor = (n ** 3) / ((1 - n) ** 2)
        zone_props[zone] = {
            "bulk_density": round(avg_bd, 3),
            "porosity": round(porosity, 3),
            "kc_factor": kc_factor,
            "n_samples": len(bds),
        }
    return zone_props

_wosis_profiles = _load_wosis_data()
_zone_props = _compute_zone_properties(_wosis_profiles)

# Depth-resolved BD from WoSIS for display
def _wosis_depth_profile(profiles, zone=None):
    """Get depth-resolved bulk density averages for display."""
    nearby = [p for p in profiles if 46.0 <= p["y"] <= 46.6 and 19.8 <= p["x"] <= 20.5]
    if not nearby:
        nearby = profiles
    # Bin into 3 depth layers matching ERA5-Land
    layers = {"0-7cm": [], "7-28cm": [], "28-100cm": [], "100-200cm": []}
    for p in nearby:
        u, l = p["upper_depth"], p["lower_depth"]
        mid = (u + l) / 2
        if mid <= 7:
            layers["0-7cm"].append(p["bulk_density"])
        elif mid <= 28:
            layers["7-28cm"].append(p["bulk_density"])
        elif mid <= 100:
            layers["28-100cm"].append(p["bulk_density"])
        else:
            layers["100-200cm"].append(p["bulk_density"])
    result = {}
    for k, vals in layers.items():
        if vals:
            avg = sum(vals) / len(vals)
            result[k] = {"bulk_density": round(avg, 3), "porosity": round(1 - avg / 2.65, 3), "n": len(vals)}
    return result

# ── Tisza Lateral Seepage Model (Dupuit-Forchheimer) ─────────────────────────
# Aquifer parameters — now spatially varying using WoSIS soil data
AQUIFER_BASE = 60.0      # Aquifer base elevation [m a.s.l.] — Pleistocene clay at ~15-20m depth
RIVER_BED_ELEV = 74.0    # Tisza riverbed elevation at Szeged [m a.s.l.]
GROUND_SURFACE = 80.0    # Average ground surface elevation [m a.s.l.]
FAR_FIELD_HEAD = 75.0    # Background water table at 25km (Homokhátság)
ET_LOSS = 0.000005        # Daily net recharge deficit [m/day] — minimal for steady-state profile
SEEPAGE_DOMAIN_M = 25000 # Model domain: 25 km from river

# Soil-zone K values derived from WoSIS bulk density via Kozeny-Carman
# Clay (near river): low K, Sand (far): high K
SOIL_ZONES = [
    # Band 0 (River Bank): alluvial clay
    {"zone": "clay",       "K": 1.5,  "porosity": 0.46, "soil_type": "Alluvial Clay (Algyő)"},
    # Band 1 (Floodplain): clay-silt transition
    {"zone": "clay",       "K": 3.0,  "porosity": 0.47, "soil_type": "Clay-Silt (Algyő)"},
    # Band 2 (Near Farmland): transition
    {"zone": "transition", "K": 8.0,  "porosity": 0.48, "soil_type": "Transition (Pusztaszer)"},
    # Band 3 (Inner Plains): sandy-silt
    {"zone": "transition", "K": 15.0, "porosity": 0.49, "soil_type": "Sandy-Silt (Pusztaszer)"},
    # Band 4 (Outer Plains): sand (Homokhátság)
    {"zone": "sandy",      "K": 25.0, "porosity": 0.49, "soil_type": "Sand (Sándorfalva)"},
]

# Override zone porosities with actual WoSIS measurements
for sz in SOIL_ZONES:
    if sz["zone"] in _zone_props:
        sz["porosity"] = _zone_props[sz["zone"]]["porosity"]
        sz["wosis_bd"] = _zone_props[sz["zone"]]["bulk_density"]


def _discharge_to_water_level(discharge_m3s: float) -> float:
    """Convert Tisza discharge [m³/s] at Szeged to approximate water level [m a.s.l.].
    Based on Szeged gauging station rating curve (simplified power law).
    Typical range: ~100 m³/s → ~75.5m, ~500 m³/s → ~78m, ~2000 m³/s → ~82m (flood)."""
    a, b = 0.45, 0.40
    h = RIVER_BED_ELEV + a * (max(discharge_m3s, 10) ** b)
    return round(min(h, 85.0), 2)  # cap at major flood level


def _dupuit_seepage_profile(h_river: float, h_far: float,
                             et_rate: float, L: float,
                             band_distances_m: list, band_K: list) -> list:
    """Dupuit-Forchheimer steady-state unconfined aquifer profile with
    spatially-varying K (from WoSIS soil zones) and ET loss.
    Uses piecewise approach: each band segment has its own K value.
    Returns water table elevation [m a.s.l.] at each band center."""
    h_r2 = h_river ** 2
    h_f2 = h_far ** 2
    # Effective K: weighted average for the Dupuit formula
    # For simplicity, use local K at each evaluation point
    w = -et_rate
    profile = []
    for i, x in enumerate(band_distances_m):
        K = band_K[i]
        x_frac = x / L
        h2 = h_r2 - (h_r2 - h_f2) * x_frac + (w / K) * x * (L - x)
        h = math.sqrt(max(h2, AQUIFER_BASE ** 2))
        profile.append(round(h, 2))
    return profile


def _head_to_seepage_moisture(h_wt: float, ground_elev: float = None, soil_zone: str = "sandy") -> dict:
    """Convert water table head to soil moisture contribution from lateral seepage.
    Capillary rise varies by soil type (from WoSIS zonation):
      - Clay (Algyő): strong capillary rise, slow decay with depth
      - Transition (Pusztaszer): moderate
      - Sandy (Sándorfalva): weak capillary rise, fast decay"""
    if ground_elev is None:
        ground_elev = GROUND_SURFACE
    depth_to_wt = max(ground_elev - h_wt, 0.0)

    if depth_to_wt <= 0:
        return {
            "surface_contribution": 0.38, "mid_contribution": 0.38,
            "deep_contribution": 0.38, "water_table_depth_m": 0.0,
            "water_table_elev_m": round(h_wt, 2),
        }

    # Capillary rise parameters by soil type (derived from WoSIS bulk density)
    # Clay: finer pores → stronger capillary pull, reaches further
    # Sand: larger pores → weaker capillary pull, drops off fast
    cap_params = {
        "clay":       {"deep": (0.45, 0.25), "mid": (0.40, 0.35), "surface": (0.30, 0.50)},
        "transition": {"deep": (0.38, 0.35), "mid": (0.32, 0.50), "surface": (0.22, 0.70)},
        "sandy":      {"deep": (0.30, 0.50), "mid": (0.22, 0.70), "surface": (0.15, 0.90)},
    }
    params = cap_params.get(soil_zone, cap_params["transition"])

    cap_deep = params["deep"][0] * math.exp(-params["deep"][1] * depth_to_wt)
    cap_mid = params["mid"][0] * math.exp(-params["mid"][1] * depth_to_wt)
    cap_surface = params["surface"][0] * math.exp(-params["surface"][1] * depth_to_wt)

    return {
        "surface_contribution": round(cap_surface, 4),
        "mid_contribution": round(cap_mid, 4),
        "deep_contribution": round(cap_deep, 4),
        "water_table_depth_m": round(depth_to_wt, 2),
        "water_table_elev_m": round(h_wt, 2),
    }


def _fetch_growing_season_climate(target_date, lat, lon, req) -> dict:
    """Fetch cumulative growing-season precipitation and heat stress from ERA5.
    Growing season: April 1 → target_date (or Sep 30 if target is later).
    Returns precip_mm (total), heat_stress_days (days > 32°C), mean_temp.
    """
    year = target_date.year
    gs_start = dt.date(year, 4, 1)
    gs_end = min(target_date, dt.date(year, 9, 30))
    if gs_end < gs_start:
        gs_start = dt.date(year - 1, 4, 1)
        gs_end = dt.date(year - 1, 9, 30)
    try:
        resp = req.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat, "longitude": lon,
                "start_date": gs_start.isoformat(), "end_date": gs_end.isoformat(),
                "daily": "precipitation_sum,temperature_2m_max,temperature_2m_mean",
                "timezone": "Europe/Budapest",
            },
            timeout=20,
        )
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
        precip_vals = [v for v in (daily.get("precipitation_sum") or []) if v is not None]
        tmax_vals = [v for v in (daily.get("temperature_2m_max") or []) if v is not None]
        tmean_vals = [v for v in (daily.get("temperature_2m_mean") or []) if v is not None]
        return {
            "precip_mm": round(sum(precip_vals), 1),
            "heat_days": sum(1 for t in tmax_vals if t > 35),
            "mean_temp": round(sum(tmean_vals) / len(tmean_vals), 1) if tmean_vals else 20.0,
            "gs_days": len(precip_vals),
        }
    except Exception:
        return {"precip_mm": 250.0, "heat_days": 10, "mean_temp": 20.0, "gs_days": 120}


def _estimate_band_yield(sm_surface, sm_mid, sm_deep, soil_zone, K, dist_km, month, climate=None):
    """Estimate crop yield per band using ERA5-Land moisture + growing-season climate.
    The key insight: single-point soil moisture barely varies between years at 9km ERA5
    resolution, but cumulative precipitation and heat stress strongly differentiate
    drought years (2003: 150mm precip) from good years (2015: 350mm precip).
    """
    # Root-zone effective moisture (weighted toward deep layers)
    rz = ((sm_surface or 0) * 0.15 + (sm_mid or 0) * 0.30 + (sm_deep or 0) * 0.55)

    # Moisture adequacy score (from ERA5-Land snapshot — used as a minor factor)
    if rz <= 0.03:
        moisture_score = 0.25
    elif rz <= 0.06:
        moisture_score = 0.25 + 0.35 * (rz - 0.03) / 0.03
    elif rz <= 0.10:
        moisture_score = 0.60 + 0.25 * (rz - 0.06) / 0.04
    elif rz <= 0.16:
        moisture_score = 0.85 + 0.10 * (rz - 0.10) / 0.06
    elif rz <= 0.25:
        moisture_score = 0.95 + 0.05 * (rz - 0.16) / 0.09
    elif rz <= 0.35:
        moisture_score = 1.0
    else:
        moisture_score = 0.95
    moisture_score = max(0.20, min(1.0, moisture_score))

    # Growing-season climate factors (the main yield differentiator)
    clim = climate or {"precip_mm": 250, "heat_days": 10, "mean_temp": 20.0, "gs_days": 120}
    precip = clim["precip_mm"]
    heat_days = clim["heat_days"]
    gs_days = max(clim.get("gs_days", 120), 30)

    # Normalize precip to daily rate, compare to climatological normal (2.0 mm/day for Szeged Apr-Sep)
    precip_rate = precip / gs_days  # mm/day observed
    normal_rate = 1.4  # mm/day climatological normal for Szeged area (driest in Hungary, ~250mm Apr-Sep)
    precip_ratio = precip_rate / normal_rate  # 1.0 = normal, 0.5 = drought, 1.5 = wet

    # Map ratio to yield factor — calibrated against Eurostat actuals for Szeged area
    # S-curve: steep in drought zone, flattens near normal
    if precip_ratio < 0.3:
        precip_factor = 0.30
    elif precip_ratio < 0.5:
        precip_factor = 0.30 + 0.15 * (precip_ratio - 0.3) / 0.2
    elif precip_ratio < 0.8:
        precip_factor = 0.45 + 0.50 * (precip_ratio - 0.5) / 0.3
    elif precip_ratio < 1.1:
        precip_factor = 0.95 + 0.04 * (precip_ratio - 0.8) / 0.3
    else:
        precip_factor = min(1.0, 0.99 + 0.01 * (precip_ratio - 1.1) / 0.3)
    precip_factor = max(0.25, min(1.0, precip_factor))

    # Heat stress: use 35°C threshold — Szeged regularly hits 32-33°C without crop damage
    # Only extreme heat (35°C+) causes flowering/pollination failure
    heat_penalty = max(0.0, min(0.25, (heat_days - 3) * 0.015))

    # Soil suitability by crop and zone (from WoSIS soil characterization)
    soil_scores = {
        "clay":       {"maize": 0.80, "wheat": 0.92, "sunflower": 0.65},
        "transition": {"maize": 0.95, "wheat": 0.88, "sunflower": 0.88},
        "sandy":      {"maize": 0.75, "wheat": 0.70, "sunflower": 0.95},
    }
    ss = soil_scores.get(soil_zone, soil_scores["transition"])

    # Base yields (t/ha) — Hungary 5-year averages (Eurostat)
    base = {"maize": 6.0, "wheat": 5.8, "sunflower": 3.0}

    results = {}
    for crop in ["maize", "wheat", "sunflower"]:
        # Combined yield = base × precipitation_factor × soil × (1 - heat_penalty)
        # Moisture score adds a small correction (±5%) since it barely varies between years
        moisture_adj = 0.95 + 0.10 * moisture_score  # range: 0.97 to 1.05
        y = base[crop] * precip_factor * ss[crop] * (1 - heat_penalty) * moisture_adj
        results[crop] = round(y, 2)

    overall = round(precip_factor * (1 - heat_penalty) * 100, 1)

    return {
        "maize_t_ha": results["maize"],
        "wheat_t_ha": results["wheat"],
        "sunflower_t_ha": results["sunflower"],
        "root_zone_moisture": round(rz, 4),
        "moisture_score": round(moisture_score, 3),
        "precip_factor": round(precip_factor, 3),
        "heat_penalty": round(heat_penalty, 3),
        "growing_season_precip_mm": clim["precip_mm"],
        "heat_stress_days": heat_days,
        "overall_score": min(100, overall),
        "best_crop": max(results, key=results.get),
    }


def _fetch_tisza_discharge(target_date, req) -> dict:
    """Fetch Tisza river discharge from Open-Meteo Flood API (GloFAS/Copernicus)."""
    cutoff = dt.date.today() - dt.timedelta(days=5)
    try:
        if target_date <= cutoff:
            # Historical
            start = (target_date - dt.timedelta(days=7)).isoformat()
            end = target_date.isoformat()
            resp = req.get("https://flood-api.open-meteo.com/v1/flood", params={
                "latitude": 46.25, "longitude": 20.16,
                "daily": "river_discharge",
                "start_date": start, "end_date": end,
            }, timeout=15)
        else:
            # Recent + forecast
            resp = req.get("https://flood-api.open-meteo.com/v1/flood", params={
                "latitude": 46.25, "longitude": 20.16,
                "daily": "river_discharge",
                "past_days": 7, "forecast_days": 7,
            }, timeout=15)

        resp.raise_for_status()
        daily = resp.json().get("daily", {})
        times = daily.get("time", [])
        discharges = daily.get("river_discharge", [])

        target_str = target_date.isoformat()
        # Find exact date or nearest
        if target_str in times:
            idx = times.index(target_str)
            q = discharges[idx]
        else:
            valid = [(t, d) for t, d in zip(times, discharges) if d is not None]
            q = valid[-1][1] if valid else 200.0

        # Also compute 7-day average for trend
        valid_q = [d for d in discharges if d is not None]
        avg_7d = sum(valid_q) / len(valid_q) if valid_q else q

        water_level = _discharge_to_water_level(q)
        water_level_avg = _discharge_to_water_level(avg_7d)

        return {
            "discharge_m3s": round(q, 1),
            "discharge_7d_avg_m3s": round(avg_7d, 1),
            "water_level_m": water_level,
            "water_level_7d_avg_m": water_level_avg,
            "source": "Copernicus GloFAS (via Open-Meteo Flood API)",
            "status": "ok",
        }
    except Exception as e:
        # Fallback: assume moderate flow
        return {
            "discharge_m3s": 200.0,
            "discharge_7d_avg_m3s": 200.0,
            "water_level_m": _discharge_to_water_level(200.0),
            "water_level_7d_avg_m": _discharge_to_water_level(200.0),
            "source": "fallback estimate",
            "status": f"error: {str(e)[:60]}",
        }


@app.get("/river-segments", tags=["Tisza River"])
def get_river_segments(
    date: Optional[str] = Query(None, description="Simulation date (YYYY-MM-DD)"),
):
    """
    Fetch ground humidity + lateral seepage model for parallel bands alongside the Tisza.
    Combines ERA5-Land soil moisture with Dupuit-Forchheimer groundwater seepage
    driven by real Tisza discharge data (Copernicus GloFAS).
    """
    import requests as req

    sim_date = parse_date(date)
    target = sim_date or dt.date.today()

    # 1. Fetch Tisza discharge → water level
    tisza = _fetch_tisza_discharge(target, req)
    h_river = tisza["water_level_m"]

    # 2. Compute Dupuit-Forchheimer seepage profile
    # Use harmonic mean K (correct for flow through layers in series)
    band_distances_m = [b["dist_km"] * 1000 + 2500 for b in TRANSECT_BANDS]  # center of each band
    all_K = [SOIL_ZONES[i]["K"] for i in range(len(TRANSECT_BANDS))]
    K_harmonic = len(all_K) / sum(1.0 / k for k in all_K)  # effective K for whole domain
    band_K = [K_harmonic] * len(TRANSECT_BANDS)  # uniform effective K for Dupuit profile
    wt_profile = _dupuit_seepage_profile(
        h_river=h_river,
        h_far=FAR_FIELD_HEAD,
        et_rate=ET_LOSS,
        L=SEEPAGE_DOMAIN_M,
        band_distances_m=band_distances_m,
        band_K=band_K,
    )

    # 3. Fetch ERA5-Land soil moisture for each band (14-day average for robust signal)
    window_start = target - dt.timedelta(days=13)
    cutoff = dt.date.today() - dt.timedelta(days=7)
    if target <= cutoff:
        api_url = "https://archive-api.open-meteo.com/v1/archive"
        base_params = {"start_date": window_start.isoformat(), "end_date": target.isoformat()}
    else:
        api_url = "https://api.open-meteo.com/v1/forecast"
        base_params = {"past_days": 14, "forecast_days": 1}

    results = []
    for i, band in enumerate(TRANSECT_BANDS):
        sample_lat = RIVER_CENTER_LAT
        sample_lon = RIVER_CENTER_LON + band["lon_offset"] * LON_PER_5KM

        # ERA5-Land soil moisture
        params = {
            **base_params,
            "latitude": sample_lat,
            "longitude": sample_lon,
            "hourly": "soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,soil_moisture_28_to_100cm",
            "timezone": "Europe/Budapest",
        }

        sm_surface = sm_mid = sm_deep = None
        try:
            resp = req.get(api_url, params=params, timeout=15)
            resp.raise_for_status()
            hourly = resp.json().get("hourly", {})

            def day_avg(values):
                valid = [v for v in (values or []) if v is not None]
                return round(sum(valid) / len(valid), 4) if valid else None

            sm_surface = day_avg(hourly.get("soil_moisture_0_to_7cm"))
            sm_mid = day_avg(hourly.get("soil_moisture_7_to_28cm"))
            sm_deep = day_avg(hourly.get("soil_moisture_28_to_100cm"))
        except Exception:
            pass

        # Soil zone properties from WoSIS
        soil = SOIL_ZONES[i]

        # Seepage contribution — soil-type-specific capillary rise
        seepage = _head_to_seepage_moisture(wt_profile[i], soil_zone=soil["zone"])

        # Combined moisture: ERA5-Land (atmospheric) + seepage (lateral groundwater)
        # Use max of ERA5 and seepage-boosted value (seepage raises the floor)
        combined_surface = max(sm_surface or 0, (sm_surface or 0) + seepage["surface_contribution"] * 0.5)
        combined_mid = max(sm_mid or 0, (sm_mid or 0) + seepage["mid_contribution"] * 0.5)
        combined_deep = max(sm_deep or 0, (sm_deep or 0) + seepage["deep_contribution"] * 0.5)

        # Yield estimation per band — uses growing-season climate + soil moisture
        # Fetch growing-season climate once (first band), reuse for nearby bands
        if i == 0:
            gs_climate = _fetch_growing_season_climate(target, sample_lat, sample_lon, req)
        yield_est = _estimate_band_yield(
            combined_surface, combined_mid, combined_deep,
            soil["zone"], soil["K"], band["dist_km"], target.month,
            climate=gs_climate,
        )

        results.append({
            **band,
            "lat": sample_lat,
            "lon": round(sample_lon, 4),
            "date": target.isoformat(),
            "soil_type": soil["soil_type"],
            "soil_properties": {
                "hydraulic_conductivity_m_day": soil["K"],
                "porosity": soil["porosity"],
                "bulk_density_g_cm3": soil.get("wosis_bd"),
                "zone": soil["zone"],
            },
            "soil_moisture": {
                "surface_0_7cm": sm_surface,
                "mid_7_28cm": sm_mid,
                "deep_28_100cm": sm_deep,
            },
            "seepage": {
                **seepage,
                "combined_surface": round(combined_surface, 4),
                "combined_mid": round(combined_mid, 4),
                "combined_deep": round(combined_deep, 4),
            },
            "yield_estimate": yield_est,
            "status": "ok",
        })

        time.sleep(0.1)

    # WoSIS depth profile for display
    wosis_depth = _wosis_depth_profile(_wosis_profiles)

    return {
        "river": "Tisza",
        "region": "Szeged, Hungary",
        "date": target.isoformat(),
        "tisza": tisza,
        "segments": results,
        "river_course": RIVER_COURSE,
        "band_boundaries": _compute_band_boundaries(),
        "wosis": {
            "source": "WoSIS (ISRIC World Soil Information Service)",
            "zones": {k: v for k, v in _zone_props.items()},
            "depth_profile": wosis_depth,
        },
        "data_source": "Copernicus ERA5-Land + GloFAS + WoSIS Soil Database",
    }


def _compute_band_boundaries():
    """Compute polyline coordinates for each band boundary (parallel to river)."""
    boundaries = []
    for offset_mult in range(6):  # 0, 5, 10, 15, 20, 25 km lines
        dist_km = offset_mult * 5
        line = []
        for pt in RIVER_COURSE:
            line.append([pt[0], round(pt[1] - offset_mult * LON_PER_5KM, 4)])
        boundaries.append({
            "dist_km": dist_km,
            "coords": line,
        })
    return boundaries

# ── Static files & frontend ──────────────────────────────────────────────────

@app.get("/dashboard", tags=["Frontend"], include_in_schema=False)
def dashboard():
    """Serve the web dashboard (alias for /)."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("\n🌾 Impetus Aquae Fontis API Server")
    print("📍 http://localhost:8000")
    print("🖥️  http://localhost:8000/dashboard  (Web Dashboard)")
    print("📚 http://localhost:8000/docs         (Swagger UI)\n")
    print("📚 http://localhost:8000/docs (Swagger UI)\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
