# District 5 Housing Affordability Dashboard
### A Data-Driven Brief for LA City Council District 5

---

## What This Is

This notebook is a **campaign-ready housing research tool** built for anyone running for — or working in — LA City Council District 5. It pulls real housing and income data directly from the U.S. Census Bureau, visualizes nearly a decade of trends in the district, and projects what those trends might look like through 2028.

No spreadsheets. No opinion polls. Just publicly available federal data, clearly visualized and ready to put in front of voters, journalists, or a policy committee.

---

## Why It Matters on the Campaign Trail

Housing affordability is consistently the top issue in local LA elections — and District 5 is no exception. Neighborhoods like Westwood, Beverly Grove, Fairfax, and Rancho Park have seen significant rent growth over the past decade while residents' incomes have not kept pace at the same rate.

This tool gives a candidate (or their policy team) the ability to:

- **Cite specific numbers** — "Rent in District 5 has risen 46% since 2015, while incomes grew only 32%" — backed by Census Bureau data, not anecdote
- **Show the trend visually** — clean, publication-quality charts that can go in mailers, presentations, or press releases
- **Make a forward-looking argument** — the Gaussian Process forecast shows where the district is headed through 2028 under current conditions, creating a compelling "if we do nothing" narrative
- **Respond to challenges** — all sources are federal, publicly auditable U.S. Census ACS data; no one can credibly dispute the methodology

---

## What the Notebook Produces

### Figure 1 — Rent Is Rising Faster Than Incomes
A side-by-side trend of median gross rent and median monthly income (2015–2023). The shaded gap between the two lines is the core visual argument. Key callout: rent up **+46%**, income up **+32%** over the period.

### Figure 2 — The Rent-to-Income Ratio
How much of a resident's monthly paycheck goes to rent, plotted against the HUD 30% cost-burden threshold. In 2023 District 5 sits at **~26.8%** — below the threshold, but rising steadily from 24.3% in 2015. The trend line is the story.

### Figure 3 — Where Are We Headed? (GP Forecast, Rent & Income to 2028)
A Gaussian Process model trained on 9 years of Census data projects median rent and income through 2028. Shaded confidence bands show the plausible range of outcomes — giving a candidate room to say "even in the best case..." or "under current conditions..."

### Figure 4 — Will We Cross the 30% Line?
The most campaign-ready chart. A direct answer to "are District 5 renters cost-burdened?" — with a dynamic callout showing the 2028 projection and whether the GP model expects the district to breach the HUD threshold. Under current trends the model projects **26.1% by 2028**, staying below the line — but the upward trajectory since 2015 is a clear warning signal.

---

## Data Sources

All data is sourced from the **U.S. Census Bureau American Community Survey (ACS) 5-Year Estimates**, accessed via the free Census API. No API key is required.

| Metric | Census Table | Years |
|---|---|---|
| Median Gross Rent | B25031 | 2015–2023 |
| Median Household Income | B19013 | 2015–2023 |

**Geographic scope:** ZIP codes 90210, 90212, 90035, 90048, 90064, 90024 — the core ZIP codes approximating District 5's boundaries. Values are unweighted means across ZIP codes.

> These ZIPs are an approximation. District 5's official boundaries do not perfectly align with ZIP code borders. For a GIS-precise analysis, cross-reference against the [LA City Council District boundary shapefiles](https://geohub.lacity.org/datasets/lahub::city-council-districts/about).

---

## How to Run It

**Requirements:** Python 3.9+ with Jupyter Notebook or VS Code with the Jupyter extension. All Python packages install automatically on first run.

```bash
# Clone or download the folder, then open the notebook
cd ciry_council
jupyter notebook city_council.ipynb
```

Or open `city_council.ipynb` directly in VS Code and click **Run All**.

The notebook will:
1. Auto-install any missing packages (`requests`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`)
2. Fetch 9 years of Census data live (requires internet; takes ~30 seconds)
3. Process the data and compute all metrics
4. Render all four publication-quality charts

**No data files to download. No accounts to create. No API keys needed.**

---

## How to Adapt It

The notebook is designed to be easy to customize without deep Python knowledge:

| What you want to change | Where to change it |
|---|---|
| Different ZIP codes | `DISTRICT_5_ZIPS` dict in Cell 5 |
| Different year range | `START_YEAR` / `END_YEAR` in Cell 5 |
| Longer forecast horizon | `FORE_YEARS` range in Cell 16 |
| Different rolling average window | `ROLLING_WINDOW` in Cell 10 |

To adapt for **a different council district**, simply update `DISTRICT_5_ZIPS` with the relevant ZIP codes and update the title strings in the chart cells.

---

## Forecasting Methodology

The forward projections (Figures 3 and 4) use **Gaussian Process Regression (GPR)** — a Bayesian machine learning method that fits a distribution of possible trend lines to the observed data, rather than assuming a rigid linear or exponential form.

The kernel used is `Constant × RBF + WhiteKernel`:
- The **RBF** component captures smooth underlying momentum (length-scale constrained to 1–20 years to prevent over-fitting a 9-year series)
- The **WhiteKernel** absorbs sampling noise inherent in Census estimates

The shaded bands in Figures 3 and 4 represent **±1σ and ±2σ posterior uncertainty** — the GP's honest accounting of how confident it is in the projection. Bands widen toward 2028 as uncertainty compounds over time.

**Treat the forecast as a scenario tool, not a point prediction.** The value for a campaign is not the exact 2028 number — it's the direction of travel and the range of plausible outcomes.

---

## Limitations & Honest Caveats

- **ZIP ≠ District boundary.** Some ZIP codes straddle the district line. A GIS-joined analysis would be more precise.
- **ACS 5-Year Estimates overlap.** Each year's estimate covers a 5-year rolling sample window, so adjacent years share respondents. This smooths year-to-year noise but means consecutive data points are not fully independent.
- **No rent control or unit-type breakdown.** The median gross rent figure mixes market-rate, rent-stabilized, and owner-occupied units. A more granular analysis would separate these cohorts.
- **9 data points is a short GP training set.** The forecast captures recent momentum well but cannot model structural shocks (e.g., a recession, new housing legislation, pandemic-era anomalies). Wide confidence bands in later forecast years reflect this honestly.
- **Income figure is household, not renter-only.** Median household income includes homeowners, who typically earn more than renters in this district. A renter-specific income series would show a worse affordability picture.

---

## Suggested Campaign Applications

- **Press Backgrounder** — export the charts (PNG, 120 DPI) and attach to a housing policy press release
- **Town Hall Slides** — Figures 1 and 4 are the most publicly legible; use them as opening slides in a housing-focused town hall
- **Policy One-Pager** — Figure 2's rent-to-income trend makes a clean single-page brief when paired with the "26.8% in 2023" callout number
- **Debate Prep** — the Census source citation makes every statistic debate-proof; opposing candidates cannot dispute federal data
- **Coalition Building** — share the notebook with tenant advocacy groups, planning commissions, or university policy programs as a transparent, reproducible research artifact

---

*Data: U.S. Census Bureau ACS 5-Year Estimates, Tables B25031 & B19013. Analysis performed in Python using open-source libraries. All code is reproducible and auditable.*

---

## SB 79 TOD Implementation (Phase 1)

Phase 1 is now implemented in [sb79_phase1_pipeline.py](sb79_phase1_pipeline.py). It bootstraps the TOD workflow by:

1. Downloading current LA Metro **bus + rail** GTFS schedule feeds
2. Computing stop-level weekday service intensity by mode
3. Assigning **preliminary** SB 79 tier labels using transparent thresholds
4. Exporting clean CSVs for downstream parcel/zoning joins

### Run Phase 1

```bash
pip install pandas requests
```

```bash
python sb79_phase1_pipeline.py
```

Optional threshold tuning:

```bash
python sb79_phase1_pipeline.py --tier1-trips 72 --tier2-trips 48
```

If your environment blocks certificate validation on external downloads:

```bash
python sb79_phase1_pipeline.py --allow-insecure-download
```

### Outputs

- `data/raw/la_metro_gtfs.zip`
- `data/raw/la_metro_gtfs_bus.zip`
- `data/raw/la_metro_gtfs_rail.zip`
- `data/interim/sb79_stop_service_summary.csv`
- `data/interim/sb79_tier_counts.csv`

### Important Note

The generated tier labels are implementation heuristics for analysis prototyping. Before publication, validate final stop eligibility and tier assignment against official SB 79 implementation maps and agency guidance (SCAG, LA City Planning, and HCD).

### Standalone Integrated Notebook

A separate notebook now exists for SB 79 + housing integration:

- `sb79_tod_housing_analysis.ipynb`

It reuses District 5 housing data logic through:

- `housing_tod_utils.py`

Run it directly in VS Code/Jupyter. It will:

1. Reuse or regenerate GTFS tier outputs
2. Build a District 5 transit proxy from ZIP centroids
3. Pull ACS housing data for District 5 ZIPs
4. Produce integrated policy metrics and charts
