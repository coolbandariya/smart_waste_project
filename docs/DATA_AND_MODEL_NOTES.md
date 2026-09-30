# Data and model notes

This repository demonstrates a waste-collection pipeline. Treat generated or sample records as illustrative unless their provenance is explicitly documented.

## Pipeline boundaries
1. Preprocessing creates model-ready features from the input time series.
2. The prediction module estimates future fill level.
3. The priority module combines operational signals into a collection score.
4. The optimizer proposes routes subject to its configured constraints.
5. The Streamlit app presents the resulting data and route.

## Interpretation
- A prediction is an estimate, not a sensor reading.
- Priority scores depend on the configured membership functions and thresholds; they are not universal urgency labels.
- A route is only as reliable as its coordinates, distance assumptions, vehicle capacities, and input validation.
- Synthetic or stale data must not be presented as live IoT telemetry.

## Validation when changing the pipeline
- Check missing values, duplicate identifiers, and out-of-range fill percentages before model use.
- Keep preprocessing and inference feature order consistent.
- Test empty inputs and cases where no feasible route exists.
- Verify every assigned stop is present in the input and vehicle capacity constraints are respected.
- Record the dataset source, generation method, and time window when publishing results.

Run the automated tests and CI workflow before merging. Do not commit credentials or private operational data.