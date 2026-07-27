# FireCast AI Research Protocol

## Dataset integrity

- Use only versioned, real observations and preserve source provenance.
- Validate schemas before feature engineering.
- Remove duplicates using an auditable rule; never remove target conflicts silently.
- Fit imputers, scalers, encoders, and feature selectors on training data only.

## Evaluation splits

- Use chronological holdout for time-indexed observations.
- Hold out geographic sites or forest regions when spatial generalization is claimed.
- Report class prevalence for every split.
- Keep a final untouched test set for publication claims.

## Metrics

Report sensitivity, specificity, precision, F1, ROC-AUC, PR-AUC, MCC, balanced accuracy,
Brier score, calibration error, confusion counts, threshold, and false-alarm cost.

## Robustness

Evaluate sensor noise/dropout and image low-light, blur, fog, and noise conditions. Report
performance degradation relative to clean validation data, not only absolute scores.

## Deployment

Record checkpoint hash, model size, runtime provider, input shape, warmup policy, mean/P50/P95
latency, throughput, power measurements where available, and software/hardware versions.

## Reproducibility

Persist configuration, seed, feature schema, data version, dependency lock, model metadata,
and generated metrics with each experiment artifact.
