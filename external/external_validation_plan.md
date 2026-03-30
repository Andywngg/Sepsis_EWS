# External Validation Plan (MIMIC-IV / eICU-CRD)

This project already includes cross-site A/B evaluation within the PhysioNet 2019 dataset. A true external validation requires running the identical pipeline on an independent ICU dataset that is not part of the challenge data. The most practical options are MIMIC-IV and eICU-CRD, both of which require credentialed access and a data use agreement.

## Minimal external validation workflow
1. Obtain access to the external dataset and export ICU time-series for the same 40 variables used by the PhysioNet 2019 challenge (hourly bins, SepsisLabel derived from Sepsis-3).
2. Convert each patient stay into a `.psv` file with the exact PhysioNet 2019 column schema.
3. Place the external `.psv` files into `sepsis_ews/data/external/` (or any folder you choose).
4. Run the evaluation script using the already trained model and the same threshold used in internal tests.

## Example command (after conversion)
```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python -m sepsis_ews.eval `
  --data-dir data\external `
  --weights outputs\utility\model.joblib `
  --medians outputs\utility\medians.json `
  --output-dir outputs\external_eval `
  --feature-set enhanced `
  --utility official `
  --alert-k 1 `
  --calibrate sigmoid
```

## Notes
This project assumes the PhysioNet 2019 schema. External validation requires a careful variable mapping and hourly aggregation that preserves the original semantics. The utility score and alert burden should be reported alongside AUROC/AUPRC to maintain clinical alignment.
