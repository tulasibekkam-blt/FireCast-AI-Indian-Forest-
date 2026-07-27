# FireCast AI

FireCast AI is a research and deployment foundation for multimodal wildfire intelligence. It deliberately has no web UI. The system is organized around reproducible data contracts, leakage-safe evaluation, calibrated risk estimates, vision/spread model adapters, explainability, and edge export.

## Current implementation

The implemented foundation currently covers three research slices and edge deployment:

- strict CSV/Parquet ingestion with schema and target validation;
- chronological or stratified train/validation/test splitting;
- preprocessing that learns only from training data;
- a comparable scikit-learn model suite;
- optional XGBoost, LightGBM, and CatBoost integrations;
- threshold selection, calibration, and publication-style binary metrics;
- artifact persistence with metadata and reproducibility information.
- Ultralytics detector training, validation, streaming inference, and latency benchmarking.
- LSTM, GRU, and causal Transformer spread forecasting with train-only scaling.
- Physics-regularized sequence loss and continuous forecast metrics.
- TorchScript/ONNX export and ONNX Runtime benchmarking.
- Grad-CAM, Integrated Gradients, SHAP, permutation importance, and counterfactual utilities.

No synthetic or dummy data is generated. A run must point at a real, labeled dataset.

## Quick start

```powershell
python -m pip install -e ".[dev,tabular]"
python -m firecast.cli train-tabular --data data/validated/wildfire_risk.parquet --target ignition --output artifacts/tabular
pytest
```

Install optional capabilities with `.[vision]`, `.[deployment]`, and `.[ingest]` as needed.

Other entry points are `validate-tabular`, `explain-tabular`, `predict-tabular`, `train-detector`,
`evaluate-detector`, `train-spread`, `benchmark-onnx`, `health`, and `drift`. All commands require real
input data or existing checkpoints; the package never creates synthetic datasets or
downloads weights implicitly.

The dataset must contain one binary target column. For time-aware evaluation, pass `--time-column` and rows are sorted before splitting. Keep site/region identifiers out of features or provide a grouped splitter in the experiment configuration to prevent spatial leakage.

## Architecture

```text
real sensors/weather/labels -> validation -> feature contract -> leakage-safe split
                                      -> preprocessing -> model comparison/tuning
                                      -> calibration + threshold policy -> metrics/artifacts
                                      -> XAI (SHAP/counterfactuals) and edge export
```

The remaining research slices are multimodal fusion, robustness benchmarks, probability-
aware alert policy evaluation, and hardware-specific TensorRT/TFLite/OpenVINO exporters.
Those depend on dataset manifests and target hardware; they should be added as adapters
rather than mixed into the existing pipelines.

## Research integrity

Every experiment should record dataset version/hash, feature schema, split policy, random seed, dependency lock, model parameters, and calibration policy. Metrics alone are not evidence of deployability: report site/time-held-out performance, calibration, latency, and false-alarm cost.
