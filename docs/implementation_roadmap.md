# FireCast AI Implementation Roadmap

## Completed foundation

- Real-data tabular wildfire-risk training, comparison, validation, calibration, explainability, inference, and artifacts.
- YOLO dataset validation, detector training/evaluation/inference, streaming, and latency benchmarking.
- LSTM/GRU/Transformer spread forecasting, train-only scaling, physics-regularized loss, evaluation, inference, and export.
- Sensor/weather ingestion, MQTT parsing, offline SQLite buffering, feature merging, and multimodal fusion.
- Alert hysteresis, cooldown, persistence, audit logging, health checks, drift monitoring, and robustness utilities.
- TorchScript/ONNX export and ONNX Runtime benchmarking.

## Remaining implementation phases

### Phase 1: Dataset and experiment completeness

- Add versioned manifests for real wildfire, satellite, CCTV, terrain, vegetation, weather, and IoT datasets.
- Add geographic and temporal holdout splitters.
- Add experiment tracking and immutable run manifests.

### Phase 2: Vision research comparison

- Add controlled adapters for YOLOv8, YOLOv11, RT-DETR, EfficientDet, Faster R-CNN, and EfficientNet.
- Normalize mAP, precision, recall, F1, latency, FPS, and model-size reporting.
- Add domain robustness evaluation across smoke, fog, rain, night, blur, and occlusion.

### Phase 3: Spread research comparison

- Add temporal convolution, cellular automata, and physics-based baselines.
- Add raster/grid targets and geospatial metrics.
- Add spatial holdout and rollout evaluation.

### Phase 4: Edge production

- Add TensorRT, TFLite, OpenVINO, and Coral export paths where hardware runtimes are present.
- Add hardware benchmark manifests and power measurements.
- Add signed model packages, rollback, and deployment health orchestration.

### Phase 5: Field validation

- Calibrate alert costs with fire-service operators.
- Run shadow-mode deployments before automated escalation.
- Report false alarms, missed detections, drift, uptime, latency, and recovery behavior.

## Publication gate

No claim of state-of-the-art performance should be made until the final test set, geographic
holdout, calibration, robustness, latency, and ablation results are archived with the model card.
