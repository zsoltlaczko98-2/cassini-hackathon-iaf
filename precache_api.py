"""Pre-cache all external API responses for Vercel serverless deployment.

Run this locally before deploying — it hits all endpoints once, causing
main.py's caching layer to save every Open-Meteo / Eurostat response to
the api_cache/ directory. On Vercel, these cached responses are served
instantly without needing external API calls.

Usage:
    python precache_api.py [--date 2025-08-15]
"""

import os
import sys
import time
import argparse
import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Ensure cache directory exists before importing main (which activates caching)
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

from main import (
    FieldConfig,
    fetch_weather_data,
    fetch_soil_moisture_profile,
    fetch_sentinel1_soil_moisture,
    estimate_ndvi,
    run_water_balance,
    generate_alerts,
    fetch_eurostat_yields,
    build_training_dataset,
    predict_current_season,
)

# Also cache server.py's API calls (GloFAS, ERA5 growing-season climate)
import server

# Ensure pretrained models are loaded
if not server._model_cache:
    server._load_pretrained_models()


def warm_cache(sim_date: dt.date):
    """Make all API calls for the given date to populate the cache."""
    cfg = FieldConfig()
    print(f"\n🔄 Pre-caching API responses for {sim_date}...\n")
    t0 = time.time()

    # 1. Weather forecast (ERA5 + Open-Meteo)
    print("  📡 Weather data (ERA5)...")
    weather = fetch_weather_data(cfg, past_days=30, forecast_days=7, sim_date=sim_date)
    print(f"     ✅ {len(weather)} days cached")

    # 2. Soil moisture profile (ERA5-Land)
    print("  🌱 Soil moisture profile (ERA5-Land)...")
    soil = fetch_soil_moisture_profile(cfg, sim_date=sim_date)
    print(f"     ✅ {len(soil.get('depths', []))} layers cached")

    # 3. Sentinel-1 SAR
    print("  📡 Sentinel-1 SAR proxy...")
    s1 = fetch_sentinel1_soil_moisture(cfg, sim_date=sim_date)
    print(f"     ✅ cached")

    # 4. Sentinel-2 NDVI (derived locally, no external API call needed)
    print("  🛰️  Sentinel-2 NDVI proxy... (local computation, no API)")
    print(f"     ✅ no caching needed")

    # 5. Water balance
    print("  💧 Water balance model...")
    states = run_water_balance(cfg, weather, past_days=30, sim_date=sim_date)
    print(f"     ✅ {len(states)} states cached")

    # 6. Alerts
    print("  🚨 Alert generation...")
    alerts = generate_alerts(states, cfg, sim_date=sim_date)
    print(f"     ✅ {len(alerts)} alerts cached")

    # 7. Yield prediction (uses cached ML model + fetches analog year data)
    print("  🤖 Yield prediction (analog years)...")
    if "maize" in server._model_cache:
        c = server._model_cache["maize"]
        pred = predict_current_season(
            cfg, c["model"], c["scaler"], c["feature_cols"],
            c["training_df"], sim_date
        )
        print(f"     ✅ prediction cached")
    else:
        print(f"     ⚠️  No maize model — run pretrain_models.py first")

    # 8. River segments (GloFAS + ERA5-Land for bands + growing-season climate)
    print("  🌊 River segments (GloFAS + ERA5-Land)...")
    try:
        import requests as req
        from server import (
            _fetch_tisza_discharge, _fetch_growing_season_climate,
            TRANSECT_BANDS, RIVER_CENTER_LAT, LON_PER_5KM, RIVER_CENTER_LON,
        )

        tisza = _fetch_tisza_discharge(sim_date, req)
        print(f"     ✅ GloFAS discharge: {tisza['discharge_m3s']:.0f} m³/s")

        # ERA5-Land for each band centroid
        for band in TRANSECT_BANDS:
            band_lat = RIVER_CENTER_LAT
            band_lon = RIVER_CENTER_LON + band["lon_offset"] * LON_PER_5KM
            band_cfg = FieldConfig()
            band_cfg.lat = band_lat
            band_cfg.lon = band_lon
            fetch_soil_moisture_profile(band_cfg, sim_date=sim_date)
        print(f"     ✅ ERA5-Land for {len(TRANSECT_BANDS)} bands")

        climate = _fetch_growing_season_climate(sim_date, RIVER_CENTER_LAT, RIVER_CENTER_LON, req)
        print(f"     ✅ Growing-season climate cached")

    except Exception as e:
        print(f"     ⚠️  River segments partial: {e}")

    # Count cached files
    n_files = len([f for f in os.listdir(CACHE_DIR) if f.endswith(".json")])
    elapsed = time.time() - t0
    total_kb = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in os.listdir(CACHE_DIR) if f.endswith(".json")) / 1024
    print(f"\n✅ Done — {n_files} API responses cached ({total_kb:.0f} KB) in {elapsed:.1f}s")
    print(f"   Cache directory: {CACHE_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="Pre-cache API responses for Vercel deployment")
    parser.add_argument("--date", type=str, default=None,
                        help="Date to cache (YYYY-MM-DD). Default: today")
    args = parser.parse_args()

    sim_date = dt.date.fromisoformat(args.date) if args.date else dt.date.today()
    warm_cache(sim_date)


if __name__ == "__main__":
    main()
