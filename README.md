# Impetus Aquae Fontis 🌊🌾

**Intelligent Agricultural Water Management Platform** — CASSINI Hackathon "Space for Water" entry

A decision-support system that uses Copernicus satellite data to forecast irrigation needs for producers near Szeged, Hungary. The system models lateral groundwater seepage from the Tisza river, predicts crop yields using machine learning, and advises when to use stored runoff water during droughts.

## How It Works

1. **Collect** — During wet seasons, runoff water is collected and stored
2. **Monitor** — Copernicus satellites track soil moisture, river levels, and vegetation health
3. **Predict** — ML model forecasts crop yield and identifies drought stress early
4. **Advise** — Producers receive alerts on when to irrigate using stored water

## Data Sources

| Source | Data | Usage |
|--------|------|-------|
| **Copernicus ERA5** | Weather reanalysis (precip, temp, ET₀) | Water balance, yield features |
| **Copernicus ERA5-Land** | Multi-depth soil moisture (0–100 cm) | Root-zone stress detection |
| **Copernicus GloFAS** | River discharge (m³/s) | Lateral seepage model input |
| **ISRIC WoSIS** | Soil bulk density profiles | Hydraulic conductivity per zone |
| **Eurostat** | 26 years of crop yield statistics | ML model training |
| **Sentinel-1/2** | SAR + NDVI/NDWI (proxy in prototype) | Vegetation & moisture indicators |

## Key Features

- **Detrended Ridge + GBM yield model** — Predicts weather-driven yield anomalies with 10.2% MAE (LOO-CV, 26 years)
- **Dupuit-Forchheimer seepage model** — Groundwater contribution from Tisza river with spatially-varying soil properties
- **7-day water balance forecast** — Daily soil moisture, ET₀, runoff opportunity detection
- **Dual-overlay Tisza River Monitor** — Interactive map with 5 transect bands (soil moisture / crop yield)
- **Analog year matching** — Projects incomplete seasons using the 5 most similar historical years

## Quick Start

### Prerequisites

```
Python 3.10+
```

### Install dependencies

```bash
pip install requests matplotlib rich numpy pandas scikit-learn fastapi uvicorn
```

### Run the dashboard

```bash
python server.py
```

Then open **http://localhost:8000** in your browser.

> ⏳ First startup takes ~60 seconds — the ML model trains on 26 years of Eurostat + ERA5 data.

### Run the CLI forecast

```bash
python main.py
```

## Dashboard

The web interface has two tabs:

- **Field Monitor** — 7-day forecast, ML yield prediction, irrigation impact analysis, alerts
- **Tisza River Monitor** — Interactive Leaflet map with seepage profile and dual overlays

## Architecture

See [`iaf_architecture_plan.md`](iaf_architecture_plan.md) for the full system architecture.

## Optional: WoSIS Soil Data

The system runs without local soil data (hardcoded fallback values are used). To enable real WoSIS measurements, download bulk density CSVs from [ISRIC WoSIS](https://www.isric.org/explore/wosis) into:

```
wosis_soil_data/
  wosis_alluvial_clay/wosis_latest_bdfiod.csv
  wosis_fluvisol_transition/wosis_latest_bdfiod.csv
  wosis_arenosol_sand/wosis_latest_bdfiod.csv
```

## Team

Built for the [CASSINI Hackathon — Space for Water](https://taikai.network/cassinihackathons/hackathons/space-for-water)

## License

MIT