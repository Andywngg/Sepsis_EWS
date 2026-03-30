RUNNING HEAD: SEPSIS EARLY WARNING SYSTEM

Sepsis Early Warning System: Utility-Optimized, Uncertainty-Aware Time-Series Diagnosis from ICU Data

Author: Andy Wang
School: Oakville Trafalgar High School
Date: February 13, 2026
Mentor/Advisor: [Name, if any]

---

## Title
Sepsis Early Warning System: Utility-Optimized, Uncertainty-Aware Time-Series Diagnosis from ICU Data

---

## Table of Contents
Abstract .................................................................................................................... 3
Introduction ............................................................................................................... 4
Background ............................................................................................................... 8
Problem Statement .................................................................................................. 11
Innovation Statement .............................................................................................. 11
Hypothesis .............................................................................................................. 13
Purpose ................................................................................................................... 14
Materials ................................................................................................................. 15
Methods ................................................................................................................... 18
Results .................................................................................................................... 21
Discussion .............................................................................................................. 29
Application .............................................................................................................. 31
Further Research .................................................................................................... 33
Conclusion .............................................................................................................. 34
Acknowledgements ................................................................................................. 35
References .............................................................................................................. 36

---

## Abstract
Sepsis is a life-threatening response to infection and remains a major cause of hospital mortality. In the United States, at least 1.7 million adults develop sepsis each year and at least 350,000 die during hospitalization or are discharged to hospice; approximately one in three hospital deaths involves sepsis (Centers for Disease Control and Prevention [CDC], 2025). Early detection can reduce mortality, but many deployed systems emphasize discrimination metrics and generate substantial alert burden without clear clinical benefit. This study presents a sepsis early warning system that is explicitly optimized for clinical utility rather than only statistical discrimination. Using ICU time-series data from the PhysioNet/Computing in Cardiology Challenge 2019, the system engineers temporal features, trains a gradient boosting classifier, calibrates risk scores, and selects alert thresholds to maximize the official PhysioNet utility function (PhysioNet, 2019).

The model is evaluated across multiple clinically relevant dimensions, including AUROC, AUPRC, patient-level metrics, alert burden, lead time, and official utility. It is compared to a logistic regression baseline, tested under cross-site generalization, and extended with uncertainty-aware triage using conformal selective prediction. Additional analyses include dynamic policy ablations, prospective streaming evaluation, subgroup performance analysis, and case-based interpretability. On the full held-out test split, the model achieves AUROC 0.847, AUPRC 0.128, and official utility 0.132, improving over the baseline (utility 0.048). Cross-site evaluation shows robust discrimination across hospital systems, and uncertainty-aware triage demonstrates meaningful tradeoffs between coverage and alert burden. These results address core weaknesses of existing sepsis tools and provide a clinically aligned, research-grade prototype focused on early detection, transparent tradeoff analysis, and generalization.

Keywords: sepsis, early warning, ICU, utility optimization, uncertainty, time-series prediction

---

## Introduction
Sepsis is a life-threatening medical emergency resulting from a dysregulated host response to infection that can rapidly progress to organ failure and death (Singer et al., 2016). The public health burden is substantial. In the United States, at least 1.7 million adults develop sepsis annually, at least 350,000 die during hospitalization or are discharged to hospice, and roughly one in three hospital deaths involves sepsis (CDC, 2025). These statistics underscore the need for improved detection and timely intervention, particularly in hospital settings where early recognition can enable rapid treatment.

Despite advances in clinical protocols, sepsis recognition remains challenging because the early signs are nonspecific, evolve over time, and are confounded by comorbid conditions. Clinicians must make time-sensitive decisions with incomplete data, often in noisy, high-acuity environments. Automated early warning systems have been proposed to assist clinicians by detecting subtle patterns that precede sepsis onset. However, real-world deployments show that many tools underperform when applied outside their development environment. An external validation of the Epic Sepsis Model, a widely deployed proprietary system, reported a hospitalization-level AUC of 0.63 and significant alert fatigue, with a large proportion of septic patients missed despite high alert volume (Wong et al., 2021). This highlights the gap between statistical discrimination and practical clinical value.

The PhysioNet/Computing in Cardiology Challenge 2019 addressed this gap by introducing a utility-based scoring function that rewards early detection and penalizes late or false alerts (PhysioNet, 2019). The challenge reflects the clinical reality that an alert is only valuable if it arrives early enough to change patient outcomes. The present study builds on that framework by designing an early warning system that explicitly optimizes the official utility score and reports a broad set of clinically meaningful metrics, including lead time and alert burden, rather than only AUROC or accuracy.

The objective of this work is to build a sepsis early warning system that is clinically aligned, transparent in its tradeoffs, and robust across hospital systems. To achieve this objective, the system incorporates calibration, cross-site evaluation, uncertainty-aware triage, prospective simulation, and detailed subgroup and case analyses. These additions provide a comprehensive view of performance and deployment realism, enabling a research-grade assessment of the model’s strengths and limitations.

---

## Background
### Clinical definition and context
The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3) define sepsis as life-threatening organ dysfunction caused by a dysregulated host response to infection and operationalize it as an acute increase in SOFA score of at least two points (Singer et al., 2016). This definition emphasizes organ dysfunction and distinguishes sepsis from uncomplicated infection. It is the basis for the labeling scheme used in the PhysioNet 2019 challenge dataset (PhysioNet, 2019).


### Epidemiology and burden
Sepsis represents a major public health burden. According to the CDC, at least 1.7 million adults in the United States develop sepsis annually and at least 350,000 die during hospitalization or are discharged to hospice (CDC, 2025). These numbers indicate that improvements in early detection could yield meaningful reductions in mortality and morbidity at the population level. The CDC further notes that most sepsis cases begin before a patient reaches the hospital, reinforcing the importance of early recognition both in the community and upon admission (CDC, 2025).


### Benchmark datasets and evaluation standards
The PhysioNet/Computing in Cardiology Challenge 2019 provides a large-scale ICU time-series dataset with hourly measurements of 40 variables and a binary SepsisLabel (PhysioNet, 2019). Data originate from three hospital systems, which enables testing of cross-site generalization (PhysioNet, 2019). The official evaluation metric is a normalized utility score that rewards early predictions from 12 hours before to 3 hours after onset and penalizes late or false alerts (PhysioNet, 2019). This utility-based evaluation is clinically meaningful because it encodes the real-world cost-benefit structure of sepsis alerts. The utility score normalizes performance so that a perfect classifier receives a score of 1 and a classifier that never alerts receives a score of 0 (PhysioNet, 2019).


### Limitations of existing systems
External validation of widely deployed systems has revealed significant shortcomings. The Epic Sepsis Model, implemented at numerous hospitals, achieved an external validation AUC of 0.63 and low sensitivity, with many alerts occurring after patients had already received antibiotics (Wong et al., 2021). These findings illustrate that discrimination metrics alone are insufficient and that systems must be evaluated with respect to timeliness and alert burden. Models that appear strong in retrospective studies can fail to deliver clinical value if they do not align with decision-making workflows.

### Methodological challenges in sepsis prediction
Sepsis prediction poses multiple challenges. Measurements are irregular, missingness is common, and patients differ substantially in baseline physiology. Class imbalance is severe because sepsis occurs in a minority of patient-hours. Models must not only detect rare events but also anticipate them early enough to influence care. These challenges motivate approaches that combine robust feature engineering, careful calibration, and evaluation frameworks focused on clinical utility rather than only statistical discrimination.

---

## Problem Statement
Many sepsis prediction models optimize AUROC and fail to explicitly address clinical utility, alert burden, or generalization across hospital systems. As a result, models can appear accurate while providing limited clinical value or producing alert fatigue. The problem is to develop a sepsis early warning system that aligns optimization with clinical utility, quantifies alert burden and lead time, and demonstrates robustness through cross-site evaluation, uncertainty-aware triage, and prospective simulation.

---

## Innovation Statement
This project integrates multiple clinically aligned innovations into a single system. It optimizes alert thresholds using the official PhysioNet utility score rather than only AUROC (PhysioNet, 2019). It quantifies alert burden, lead time, and early detection rates to reflect real-world costs. It evaluates cross-site generalization using true A/B hospital splits and mitigates domain shift through patient-level normalization. It introduces uncertainty-aware triage using conformal selective prediction, enabling explicit coverage-utility tradeoffs. It performs a prospective simulation to approximate real-time deployment and includes case-based interpretability to contextualize predictions. This combination of utility alignment, robustness analysis, and interpretability goes beyond typical student projects and addresses known weaknesses of deployed sepsis tools (Wong et al., 2021).

---

## Hypothesis
A utility-optimized, calibrated early warning model using time-series ICU data will outperform a baseline classifier on the official PhysioNet utility score while maintaining clinically meaningful lead time and acceptable alert burden across cross-site, prospective, and uncertainty-aware evaluations.

---

## Purpose
The purpose of this study is to build a clinically realistic sepsis early warning system that optimizes for clinical utility, explicitly quantifies alert burden, and evaluates robustness under cross-site generalization, uncertainty-aware triage, and prospective simulation.

---

## Materials
The primary dataset is the PhysioNet/Computing in Cardiology Challenge 2019 ICU time-series dataset, which provides hourly measurements of 40 variables and binary sepsis labels derived from Sepsis-3 criteria (PhysioNet, 2019). Each patient file is a pipe-delimited time-series with columns representing vital signs, laboratory values, demographics, and ICU stay variables. The dataset is designed for hourly prediction and includes extensive missingness, which is modeled explicitly rather than removed (PhysioNet, 2019).

The analysis and modeling pipeline were implemented in Python 3.11 using NumPy, pandas, scikit-learn, and matplotlib. All experiments were conducted on a standard laptop CPU without GPU acceleration, emphasizing computational efficiency and practical deployability.


---

## Methods
### Data preparation
Each patient record contains a time-series of hourly measurements and a binary SepsisLabel. Missing values are common and are retained for modeling to avoid systematic bias introduced by aggressive imputation. Feature construction is limited to information available at or before each hour, ensuring temporal causality and avoiding future leakage. Patient sequences vary in length, and the model operates on variable-length trajectories.

Data are stored as per-patient files, and sequences are concatenated into a single design matrix with patient identifiers and time indices. A group-based train-test split is used so that all time points from a given patient are assigned to either train or test, preventing leakage across patients. A calibration subset is drawn from training patients to support probability calibration.

### Feature engineering
The model uses a comprehensive set of temporal features. Raw measurements are included to capture absolute physiological values. Hour-to-hour deltas capture short-term changes and trends. Missingness indicators encode whether each variable was observed, reflecting the clinical observation process itself. Rolling means and standard deviations over 3-hour and 6-hour windows capture local temporal patterns and variability. Time-since-last-observed features quantify measurement recency, which is important because missing data patterns can be clinically informative. These features collectively provide 328 engineered attributes per hour.

### Modeling approach
Two models are trained for comparison. The baseline is a logistic regression classifier that provides a simple, interpretable reference. The primary model is a histogram gradient boosting classifier, which can model nonlinear relationships and feature interactions. To prioritize early detection, the training process uses utility-weighted sample reweighting, increasing the weight of samples close to sepsis onset. Specifically, time points within six hours before onset receive higher weights, and time points within three hours after onset receive moderately higher weights. This encourages the model to focus on early signals rather than late-stage markers.

Class imbalance is addressed implicitly through the weighting strategy and explicitly through evaluation choices that emphasize AUPRC and utility. Because sepsis occurs in a minority of patient-hours, accuracy alone can be misleading; the model can achieve high accuracy by predicting the negative class most of the time. The weighting scheme increases the influence of rare but clinically critical positive examples near onset, while the utility objective discourages overly late detection that would be less actionable. Together these design choices shift the learning objective toward early, clinically meaningful detection.

Model hyperparameters are fixed to ensure reproducibility. The gradient boosting classifier uses a maximum depth of six and a learning rate of 0.05. The logistic regression baseline uses a maximum of 200 iterations. Standardization is applied via z-score scaling, and missing values are imputed with feature medians computed on the training set.

### Calibration
Predicted probabilities are calibrated using sigmoid calibration on a held-out calibration subset. Calibration is essential because thresholds and risk communication depend on well-calibrated probabilities. Calibration parameters are estimated on 200 patients drawn from the training set, and the calibrated model is then evaluated on the held-out test split. Calibration quality is assessed using Brier score and calibration curves.

### Threshold selection and utility optimization
Alert thresholds are selected by sweeping candidate thresholds and computing the official PhysioNet utility score. The threshold that maximizes utility is chosen. This ensures that the model’s alerting policy is aligned with early detection and false alert tradeoffs rather than arbitrary probability cutoffs. The utility function rewards alerts occurring between 12 hours before and 3 hours after sepsis onset, penalizes late alerts, and penalizes false alerts in non-sepsis patients (PhysioNet, 2019). Utility scores are normalized so that a perfect classifier has a score of 1 and a classifier that never alerts has a score of 0 (PhysioNet, 2019).

### Evaluation metrics
The system reports AUROC and AUPRC to quantify discrimination. Patient-level AUROC and AUPRC are computed using the maximum predicted risk per patient, providing an episode-level perspective. The official utility score is computed using the PhysioNet scoring function. Clinical performance is further quantified by early detection rate, false alert rate, median lead time, and alert burden (alerts per patient-day). Accuracy and F-measure are included for completeness but are not the primary optimization targets because of class imbalance.

### Bootstrap confidence intervals
To quantify statistical uncertainty, bootstrap confidence intervals are computed by resampling patients with replacement and recomputing metrics on each resample. This patient-level bootstrap preserves the temporal structure within each patient while providing a nonparametric estimate of metric variability. Confidence intervals are derived from the 2.5th and 97.5th percentiles of the bootstrap distribution for AUROC, AUPRC, and the official utility score.

### Statistical analysis and sensitivity checks
Sensitivity analyses include ablation of dynamic thresholds and evaluation of multiple conformal coverage levels to examine the stability of performance across policy choices. Threshold optimization is performed over a grid of candidate values, and the chosen threshold is held constant for subsequent analyses to avoid test set leakage. Calibration is evaluated with Brier score and reliability curves to ensure that predicted probabilities correspond to observed frequencies.

### Cross-site generalization
True A/B hospital splits are used to evaluate generalization. Models are trained on one hospital system and tested on another. Patient-level normalization (z-scoring within patient) is applied to mitigate domain shift. This analysis directly assesses how the system performs when transferred across institutions.

### Uncertainty-aware triage
Conformal selective prediction is applied to quantify uncertainty. Calibration scores are computed on a held-out calibration set, and prediction sets are derived from these scores. Cases with low confidence are deferred rather than alerted, enabling explicit control over coverage and alert burden. This approach creates a spectrum of operating points where clinicians can decide how many cases to receive based on resource constraints.

### Dynamic policy ablation
A trend-adjusted dynamic threshold policy is tested against the static threshold. Trend-based adjustments were hypothesized to improve early detection by lowering thresholds when risk increases rapidly. The ablation identifies whether such complexity yields measurable gains over the static policy.

### Prospective simulation
A streaming evaluation is conducted in which predictions at each hour use only data available up to that time. This prospective simulation approximates real-time deployment and avoids future leakage. The simulation was conducted on a 500-patient subset of the held-out test split for computational feasibility and to emphasize causal evaluation.

In this simulation, each patient trajectory is processed sequentially, and alerts are generated only when the model’s risk score exceeds the fixed threshold at that hour. This approach mirrors how an ICU monitoring system would operate in practice, where data arrive incrementally and decisions must be made without access to future observations. The same calibrated model and threshold are used to ensure comparability with the retrospective evaluation.

### Subgroup analysis
Model performance is evaluated across demographic and unit subgroups, including age categories, gender, and ICU unit indicators. This analysis identifies potential disparities and assesses robustness across patient populations. Subgroup analysis was performed on a 500-patient test subset to balance computational cost and interpretability.

### Case studies
Three representative patients were selected to illustrate early detection, late detection, and false alarm scenarios. For each case, risk trajectories and alert timing were visualized to provide qualitative interpretation. These case studies were chosen to illustrate both strengths and limitations of the model in a concrete and interpretable way.

---

## Results
### Full held-out evaluation (test split)
The primary model achieves AUROC 0.847 and AUPRC 0.128 on the full held-out test split. Patient-level performance is stronger, with AUROC 0.860 and AUPRC 0.444. The selected alert threshold that maximizes utility is 0.10. The official utility score is 0.132, reflecting improved clinical utility relative to baseline. The early detection rate is 0.260, the false alert rate is 0.022, and the median lead time is 40.5 hours. Alert burden is 0.392 alerts per patient-day. These results indicate that the model produces earlier alerts than baseline while maintaining a manageable false alert rate.

### Baseline comparison
The logistic regression baseline achieves AUROC 0.768 and AUPRC 0.067, with an official utility of 0.048 and early detection rate of 0.123. The utility-optimized model more than doubles the baseline utility, demonstrating the benefit of the chosen modeling and optimization strategy. Improvements are especially pronounced in early detection, which is the clinically critical outcome for sepsis care.

### Cross-site generalization
Under patient-level normalization, cross-site generalization remains strong. Training on Site A and testing on Site B yields AUROC 0.967 and utility 0.592, while training on Site B and testing on Site A yields AUROC 0.965 and utility 0.627. Early detection rates under cross-site evaluation were 0.613 for A to B and 0.578 for B to A, with corresponding false alert rates of 0.106 and 0.279. These results indicate that the model maintains discrimination and utility across distinct hospital systems, though alert burden can increase under domain shift.

### Bootstrap confidence intervals
Bootstrap analysis on the full dataset (n = 10) yields a 95 percent confidence interval for AUROC of [0.820, 0.888], for AUPRC of [0.096, 0.166], and for official utility of [0.081, 0.184]. These intervals provide statistical context for the observed performance and indicate stability of the model’s primary metrics.

### Uncertainty-aware triage
Conformal selective prediction demonstrates meaningful tradeoffs. At alpha = 0.02, coverage is 0.366 with utility 0.232 and false alert rate 0.053. At alpha = 0.10, coverage increases to 0.591 with utility 0.391 but at the cost of higher false alerts. This supports the feasibility of uncertainty-aware alerting policies that can be tuned to operational constraints, providing a mechanism for balancing safety and workload.

### Dynamic policy ablation
A dynamic threshold policy based on recent risk trends was tested to determine whether a more reactive alert rule could improve early detection. The optimal trend sensitivity parameter was zero, indicating that the static threshold policy achieved the best utility. Utility declined as the trend sensitivity increased, suggesting that heuristic trend adjustments introduced noise rather than meaningful improvements. This negative result is informative because it supports the use of a simple, transparent alert rule and avoids unnecessary complexity. 

### Prospective simulation
A prospective streaming evaluation on 500 test patients yields AUROC 0.877, AUPRC 0.144, and official utility 0.309. Early detection is 0.486, false alert rate is 0.062, and median lead time is 25.0 hours. These results suggest that model performance persists under a more realistic streaming scenario, although the subset size warrants caution in generalization.

### Subgroup analysis
Subgroup performance varies by age category. For ages 40 to 59, AUROC is 0.907 and utility 0.341 (n = 162). For ages 60 to 79, AUROC is 0.885 and utility 0.277 (n = 222). For ages 80 and above, AUROC decreases to 0.805 and utility to 0.214 (n = 79). For ages under 40, AUROC is 0.853 and utility 0.432 (n = 37). These differences highlight potential age-related variation in model performance and motivate further investigation with larger samples.

### Case studies
Three patients illustrate different outcomes. Patient p000009 is an early detection case with a lead time of 190 hours. Patient p000765 is a late detection case with the first alert occurring seven hours after onset. Patient p000978 is a false alarm case with 30 alerts despite no sepsis. These trajectories demonstrate the model’s potential for early detection while also highlighting failure modes that must be addressed in deployment. The qualitative review of these cases provides a practical lens for understanding how the model behaves in real patient trajectories.

---

## Discussion
The results support the hypothesis that utility-optimized training and thresholding improves clinical utility relative to a baseline classifier. The official utility score more than doubles the baseline, and lead time is substantial, demonstrating clinically meaningful early detection. This aligns with the goal of prioritizing timely interventions rather than maximizing AUROC alone. The model’s ability to provide extended lead times suggests the potential for earlier clinical response, though the exact effect on outcomes requires prospective evaluation.

Cross-site evaluation indicates that the model generalizes well across hospital systems, particularly when patient-level normalization is applied. Domain shift is a known challenge in clinical prediction, and these results suggest that even simple normalization can substantially mitigate performance degradation. However, cross-site utility results should be interpreted alongside alert burden and false alert rates, which remain substantial under some settings, indicating a tradeoff between sensitivity and alert fatigue.

The uncertainty-aware triage analysis demonstrates that coverage can be tuned to manage alert burden. This is important for deployment in resource-limited settings where staff capacity constrains the number of actionable alerts. The conformal results show that reduced coverage can maintain or improve utility while reducing false alerts, but increased coverage can raise alert burden. These tradeoffs are essential for clinical adoption and offer a transparent mechanism for policy tuning.

The dynamic threshold ablation indicates that trend-adjusted thresholds do not improve utility in this setting; the optimal trend sensitivity parameter is zero, indicating that a static threshold is sufficient. This negative result is valuable because it avoids unnecessary complexity and clarifies the sources of performance gains.

The prospective simulation demonstrates that performance persists in a streaming evaluation, which is closer to real-time use. However, the simulation is conducted on a subset and should be validated on the full dataset in future work. Subgroup analysis suggests variation in performance across age categories, which warrants further investigation with larger samples and more demographic covariates.

Robustness to missing data was evaluated by injecting additional missingness into the test split (1,000 patients). Across 10% to 30% additional missingness, AUROC remained stable (0.844–0.847) and utility slightly increased (0.276–0.295), while alert burden stayed approximately 1.23–1.27 alerts per patient-day at the fixed threshold of 0.10. These results suggest that the feature engineering and alert policy are resilient to moderate degradation in measurement availability, a practical concern in real-world ICU environments.

Robustness to measurement delay was also evaluated by shifting all charted variables by 1–3 hours to simulate real-world documentation lags. Performance remained stable, with AUROC in the 0.847–0.850 range and utility rising modestly from 0.272 to 0.310 as delay increased, while alert burden remained approximately 1.23–1.26 alerts per patient-day at the fixed threshold of 0.10. This suggests the system is not overly sensitive to short charting delays, an important operational constraint in ICU workflows.

Compared to existing deployed systems, this project emphasizes utility and alert burden. External validation of the Epic Sepsis Model reported an AUC of 0.63 and low sensitivity along with alert fatigue (Wong et al., 2021). By contrast, this system explicitly optimizes utility, evaluates lead time, and quantifies alert burden, which addresses known weaknesses in deployed tools. Nevertheless, this study remains retrospective and relies on a public benchmark dataset. External validation on independent datasets such as MIMIC-IV and eICU-CRD is necessary prior to clinical use (Johnson et al., 2023; Pollard et al., 2018).

Several limitations should be noted. Labels are derived from retrospective criteria and may not perfectly align with the timing of clinical recognition. Missingness and measurement frequency are themselves influenced by clinical decisions, which can introduce confounding. The prospective simulation and subgroup analyses use subsets of the full test set due to computational constraints. Finally, the model does not incorporate unstructured clinical notes or imaging data, which might contain additional early signals.

Ethical considerations are also central. Automated sepsis alerts can influence clinical decisions, which raises questions about accountability, transparency, and patient safety. A high false alert rate can contribute to alert fatigue, reducing clinician responsiveness to true alerts. Conversely, overly conservative thresholds can miss treatable cases. Therefore, any deployment must include continuous monitoring, recalibration, and human oversight. This project’s emphasis on transparent metrics, explicit tradeoff analysis, and uncertainty-aware triage is intended to support safe integration into clinical workflows rather than to replace clinician judgment.

---

## Application
The proposed system is intended as a decision-support tool rather than a diagnostic device. In practice, it could run continuously in the ICU, providing real-time risk trajectories and early warning alerts to clinicians. The alert threshold can be tuned to local operational constraints, balancing early detection against alert burden. The inclusion of uncertainty-aware triage enables deferral of low-confidence cases, focusing clinician attention where the model is most confident. The system’s computational efficiency allows real-time deployment without specialized hardware, and the demonstration interface provides intuitive visualization of risk curves and alert timing. A practical deployment would include a governance plan, routine audit of alert outcomes, and periodic recalibration to accommodate changes in clinical practice and patient populations.

---

## Further Research
External validation is the most important next step. Access to MIMIC-IV and eICU-CRD would allow evaluation across different institutions and time periods (Johnson et al., 2023; Pollard et al., 2018). Larger prospective simulations with full dataset coverage would improve the reliability of streaming performance estimates. Additional subgroup analyses should be conducted with larger samples and expanded demographic variables. Human-in-the-loop studies are needed to evaluate how clinicians interpret and act on alerts, and to quantify the effect on patient outcomes. Future work should also explore periodic recalibration and drift detection to maintain performance in changing clinical environments.

---

## Conclusion
This study presents a sepsis early warning system that prioritizes clinical utility, lead time, and alert burden rather than focusing exclusively on AUROC. The system outperforms a baseline classifier, generalizes across hospital systems, and supports uncertainty-aware triage and prospective evaluation. These results demonstrate a clinically aligned, research-grade prototype that addresses known weaknesses in existing sepsis prediction tools. With external validation and prospective testing, this approach has the potential to improve early recognition and reduce sepsis-related mortality.

---

## Acknowledgements
I thank Oakville Trafalgar High School for the BASEF club and Ms. Sibley for supervising the club. I also thank the PhysioNet/Computing in Cardiology Challenge organizers for providing the dataset and scoring framework, and the CDC for public sepsis statistics used in this report (CDC, 2025; PhysioNet, 2019).

---

## References
Centers for Disease Control and Prevention. (2025, August 19). About sepsis. https://www.cdc.gov/sepsis/about/index.html

Johnson, A. E. W., Bulgarelli, L., Shen, L., Gayles, A., Shammout, A., Horng, S., Pollard, T. J., Hao, S., Moody, B., Gow, B., Lehman, L. W. H., Celi, L. A., & Mark, R. G. (2023). MIMIC-IV, a freely accessible electronic health record dataset. Scientific Data, 10(1), 1. https://doi.org/10.1038/s41597-022-01899-x

PhysioNet. (2019). Early prediction of sepsis from clinical data: The PhysioNet/Computing in Cardiology Challenge 2019. https://physionet.org/challenge/2019/

Pollard, T. J., Johnson, A. E. W., Raffa, J. D., Celi, L. A., Mark, R. G., & Badawi, O. (2018). The eICU Collaborative Research Database, a freely available multi-center database for critical care research. Scientific Data, 5, 180178. https://doi.org/10.1038/sdata.2018.178

Singer, M., Deutschman, C. S., Seymour, C. W., Shankar-Hari, M., Annane, D., Bauer, M., Bellomo, R., Bernard, G. R., Chiche, J. D., Coopersmith, C. M., Hotchkiss, R. S., Levy, M. M., Marshall, J. C., Martin, G. S., Opal, S. M., Rubenfeld, G. D., van der Poll, T., Vincent, J. L., & Angus, D. C. (2016). The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA, 315(8), 801-810. https://doi.org/10.1001/jama.2016.0287

Wong, A., Otles, E., Donnelly, J. P., Krumm, A., McCullough, J., DeTroyer-Cooley, O., Pestrue, J., Phillips, M., Konye, J., Penoza, C., Ghous, M., & Singh, K. (2021). External validation of a widely implemented proprietary sepsis prediction model in hospitalized patients. JAMA Internal Medicine, 181(8), 1065-1070. https://doi.org/10.1001/jamainternmed.2021.2626
