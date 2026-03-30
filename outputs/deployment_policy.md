# Deployment Policy Summary

This system is intended for decision support and should be deployed with explicit alert governance. A practical deployment would calibrate probabilities on recent local data, set a target alert budget (alerts per patient-day), and select the threshold that maximizes utility while meeting the budget. The threshold should be reviewed periodically as patient mix and clinical practice evolve.

Operational monitoring should include AUROC, AUPRC, official utility, early detection rate, false alert rate, and alert burden. Performance drift can be detected by tracking these metrics monthly and recalibrating when Brier score or alert burden deviates materially. If uncertainty-aware triage is enabled, coverage targets should be set based on staffing capacity, and deferred cases should be routed to clinician review rather than ignored.

No automated alert should directly trigger treatment. Alerts should be presented with a risk trajectory and time-to-onset context to support clinician interpretation. A human-in-the-loop workflow and periodic outcome audits are required to ensure safety and avoid alert fatigue.
