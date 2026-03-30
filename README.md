# Sepsis Early-Warning System (Time-Series Diagnosis)

This project builds a clinically realistic **early sepsis warning** system from ICU time-series data.
It is **not image-based**; it uses vitals and labs to predict sepsis **hours before onset**.

Key idea: diagnose only when the model is confident **and early**, and report
**time-to-detection** and **false-alarm rates** (not just AUROC).

## Dataset (PhysioNet Challenge 2019)
Download the training set from PhysioNet. The most reliable path is the public S3 mirror:

1) Install AWS CLI (Windows):
   - https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
2) Download the two training sets (A + B) into one folder:
   - `aws s3 sync --no-sign-request s3://physionet-open/challenge-2019/1.0.0/training/training_setA/ data\train`
   - `aws s3 sync --no-sign-request s3://physionet-open/challenge-2019/1.0.0/training/training_setB/ data\train`

Place patient files in:
- `sepsis_ews/data/train/` (all `.psv` files)

If you prefer the website (requires free account + data use agreement):
- https://physionet.org/content/challenge-2019/1.0.0/

### (Optional) Synthetic data for quick testing
If you don't have the dataset yet, you can generate a small synthetic dataset:
- `.\.venv\Scripts\python scripts\make_synthetic.py --output-dir data\synth --patients 50 --hours 24`

## Quick start
1) Create venv + install deps:
   - `python -m venv .venv`
   - `.\.venv\Scripts\pip install -r requirements.txt`
   - `set PYTHONPATH=src`

2) Train baseline model:
   - `.\.venv\Scripts\python -m sepsis_ews.train --data-dir data\train --model logreg --output-dir outputs\baseline`

3) Train utility-weighted model (stronger early detection):
   - `.\.venv\Scripts\python -m sepsis_ews.train --data-dir data\train --model hgb --utility-weighted --feature-set enhanced --utility official --output-dir outputs\utility`
   - (with patient normalization) `.\.venv\Scripts\python -m sepsis_ews.train --data-dir data\train --model hgb --utility-weighted --feature-set enhanced --utility official --patient-normalize --output-dir outputs\utility_norm`

4) Evaluate on held-out test split (+ quality tradeoff):
   - `.\.venv\Scripts\python -m sepsis_ews.eval --data-dir data\train --weights outputs\utility\model.joblib --medians outputs\utility\medians.json --feature-set enhanced --utility official --quality-report --output-dir outputs\eval`
   - (with calibration) `.\.venv\Scripts\python -m sepsis_ews.eval --data-dir data\train --weights outputs\utility\model.joblib --medians outputs\utility\medians.json --feature-set enhanced --utility official --calibrate sigmoid --calibration-fraction 0.1 --output-dir outputs\eval`
   - (with patient normalization) `.\.venv\Scripts\python -m sepsis_ews.eval --data-dir data\train --weights outputs\utility_norm\model.joblib --medians outputs\utility_norm\medians.json --feature-set enhanced --utility official --patient-normalize --output-dir outputs\eval_norm`

5) Generate a brief report:
   - `.\.venv\Scripts\python scripts\make_brief.py --metrics outputs\eval\metrics.json --baseline outputs\baseline_eval\metrics.json --output outputs\isef_brief.md`
6) Generate a one-page PDF (optional):
   - `.\.venv\Scripts\python scripts\make_onepager.py --metrics outputs\eval\metrics.json --utility-plot outputs\eval\utility_curve.png --quality-plot outputs\eval\quality_tradeoff.png --calibration-plot outputs\eval\calibration_curve.png --leadtime-plot outputs\eval\lead_time_hist.png --output outputs\isef_onepager.pdf`
7) Baseline vs improved comparison table:
   - `.\.venv\Scripts\python scripts\compare_models.py --baseline outputs\baseline_eval\metrics.json --improved outputs\eval\metrics.json --output-csv outputs\comparison.csv --output-md outputs\comparison.md`

## Optional: quick hyperparameter sweep (small subset)
- `.\.venv\Scripts\python scripts\tune_hgb.py --data-dir data\train --output-dir outputs\tuning --max-patients 500 --utility official`

## Optional: calibration sweep (tradeoff utility vs probability accuracy)
- `.\.venv\Scripts\python scripts\calibration_sweep.py --data-dir data\train --weights outputs\utility\model.joblib --medians outputs\utility\medians.json --output-dir outputs\calibration --max-patients 500 --utility official`

## Optional: error analysis + model card
- `.\.venv\Scripts\python scripts\error_analysis.py --data-dir data\train --weights outputs\utility\model.joblib --medians outputs\utility\medians.json --metrics outputs\eval\metrics.json --output-dir outputs\analysis --max-patients 500`
- `.\.venv\Scripts\python scripts\make_model_card.py --metrics outputs\eval\metrics.json --error-analysis outputs\analysis\error_analysis.json --cross-site outputs\cross_site\cross_site_summary.json --calibration-best outputs\calibration\best_calibration.json --output outputs\model_card.md`

## Optional: bootstrap confidence intervals
- `.\.venv\Scripts\python scripts\bootstrap_ci.py --data-dir data\train --weights outputs\utility\model.joblib --medians outputs\utility\medians.json --output outputs\bootstrap_ci.json --max-patients 5000 --calibrate sigmoid --n-bootstrap 200`

## Optional: selective prediction (defer policy)
- `.\.venv\Scripts\python scripts\defer_policy.py --data-dir data\train --weights outputs\utility\model.joblib --medians outputs\utility\medians.json --output-dir outputs\defer_policy --max-patients 5000 --calibrate sigmoid`

## Optional: alert policy analysis (sensitive vs conservative)
- `.\.venv\Scripts\python scripts\policy_analysis.py --data-dir data\train --weights outputs\utility\model.joblib --medians outputs\utility\medians.json --output-dir outputs\policy --max-patients 5000 --calibrate sigmoid`

## Demo (live presentation)
Install demo dependencies:
- `.\.venv\Scripts\pip install -r requirements_demo.txt`

Run the demo app:
- `streamlit run demo_app\app.py`

## Cross-site generalization (Train A → Test B, and vice versa)
Option A (true A/B sets):
- `aws s3 sync --no-sign-request s3://physionet-open/challenge-2019/1.0.0/training/training_setA/ data\trainA`
- `aws s3 sync --no-sign-request s3://physionet-open/challenge-2019/1.0.0/training/training_setB/ data\trainB`
- `.\.venv\Scripts\python scripts\cross_site_eval.py --data-dir-a data\trainA --data-dir-b data\trainB --output-dir outputs\cross_site --utility official --feature-set enhanced --model hgb`
- (with target calibration) `.\.venv\Scripts\python scripts\cross_site_eval.py --data-dir-a data\trainA --data-dir-b data\trainB --output-dir outputs\cross_site_cal --utility official --feature-set enhanced --model hgb --calibrate-target sigmoid --calibration-fraction 0.1 --calibration-max-patients 200`
- (with patient normalization) `.\.venv\Scripts\python scripts\cross_site_eval.py --data-dir-a data\trainA --data-dir-b data\trainB --output-dir outputs\cross_site_norm --utility official --feature-set enhanced --model hgb --patient-normalize`

Note: patient-level normalization helps mitigate domain shift across sites and substantially improves cross-site utility.

Option B (proxy A/B split from combined folder):
- `.\.venv\Scripts\python scripts\cross_site_eval.py --combined-dir data\train --split-n 20336 --output-dir outputs\cross_site --utility official --feature-set enhanced --model hgb`

## Outputs
- `outputs/metrics.json` with AUROC/AUPRC + early-warning stats
- `outputs/eval/calibration_curve.png` (probability calibration)
- `outputs/eval/lead_time_hist.png` (lead time histogram)
- `outputs/isef_brief.md` (ISEF-ready summary)

## Notes
This is a research prototype and not a clinical diagnostic tool.
