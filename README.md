````markdown
# LPDG x RGM Innovation Challenge 2026

## Machine Learning Solution

This repository contains my solution for the **NEXORA 2026 LPDG x RGM Innovation Challenge**.

The objective is to help the field team decide **which 15 gateways should be prioritised for a site visit each week**, instead of relying only on spreadsheets or manual judgement.

I selected the **Machine Learning** track for Part 2 and built a cost-aware ranking pipeline using historical gateway telemetry and meter-read outcomes.

---

## Problem

The field team can perform only **15 site visits per week**.

There are two important operational costs:

- **€380** for a wasted site visit.
- **€600 per week** when a broken gateway is left unattended.

Therefore, the goal is not simply to maximise classification accuracy.

The main question is:

> **Which 15 gateways should the field team visit first to reduce operational cost?**

---

## Approach

The solution follows this workflow:

```text
Historical Data
      ↓
Data Cleaning & Gateway ID Normalisation
      ↓
Historical Telemetry Feature Engineering
      ↓
7 / 14 / 28 Day Feature Windows
      ↓
Logistic Regression
      ↓
Gateway Risk Scores
      ↓
Deterministic Ranking
      ↓
Top 15 Gateways per Week
      ↓
predictions.csv
      ↓
Submission Validation
````

---

## Data Used

The solution uses the supplied challenge datasets:

* Gateway telemetry
* Gateway master data
* Meter-read success data

The telemetry data is aggregated into historical features before each prediction week.

Field-visit outcomes and engineer review categories were **not used as the prediction target**, because field visits are intervention-driven and therefore do not provide an unbiased representation of gateway failure.

---

## Machine Learning Model

### Target

I defined a gateway-week as a **severe failure** when the meter-read failure rate is greater than 30%.

```text
failure_rate = 1 - meters_read / meters_expected
```

The target is therefore:

```text
1 → failure_rate > 30%
0 → otherwise
```

The 30% threshold was selected after comparing operational costs for different thresholds.

| Target threshold | Validation cost |
| ---------------- | --------------: |
| >10%             |        €155,400 |
| >20%             |        €142,800 |
| **>30%**         |    **€129,600** |

The >30% threshold produced the lowest cost in the tested validation period.

This is an operational modelling choice, not a claim that 30% is a universal definition of a faulty gateway.

---

## Feature Engineering

Historical telemetry features are created using:

* **7-day window**
* **14-day window**
* **28-day window**

The final model contains **69 features**.

The features capture signals such as:

* Offline duration
* Disconnection count
* Reboot count
* Reboot duration
* Online duration
* Load
* Uptime
* Signal quality
* Importance indicators
* Telemetry coverage

For every prediction week, the feature pipeline uses information available **before that week**.

The target week's telemetry is not used to construct its prediction features.

---

## Final Model

The final model is **Logistic Regression** with:

* Median imputation
* Standard scaling
* Balanced class weights
* `max_iter=2000`
* Deterministic `random_state=42`

The model was selected because it provides a good balance between performance, interpretability and ease of modification during the live evaluation.

I also experimented with additional trend features. Although they improved some fixed-split metrics, they performed worse in the temporal walk-forward cost evaluation.

| Model       | Walk-forward cost |
| ----------- | ----------------: |
| Base model  |      **€208,300** |
| Trend model |          €211,240 |

Therefore, I retained the simpler 69-feature model.

---

## Ranking Strategy

For every prediction week:

1. All available gateways are scored.
2. Scores are sorted in descending order.
3. The top 15 gateways are selected.
4. `gateway_id` is used as a deterministic secondary sort key when scores are tied.
5. Ranks 1–15 are assigned.
6. A short operational reason is generated from recent telemetry signals.

This ensures that the same input produces a deterministic ranking.

---

## Cost-Based Evaluation

The main evaluation criterion is the **operational cost**, rather than accuracy or AUC alone.

I used an **8-week temporal holdout** to compare the ML ranking against the supplied 3-sigma baseline.

| Approach             | Total operational cost |
| -------------------- | ---------------------: |
| **Machine Learning** |           **€129,600** |
| 3-sigma baseline     |               €130,800 |
| **ML improvement**   |     **€1,200 (0.92%)** |

Across the complete 8-week validation period, the ML ranking produced a lower total operational cost than the supplied baseline.

However, the performance varied between individual weeks, so this result should not be interpreted as a guarantee that ML will outperform the baseline every week.

---

## Validation and Generalisation

I used validation strategies intended to avoid relying only on random row-level splits.

The experiments included:

* Temporal holdout evaluation
* Unseen-gateway evaluation
* Different time-regime checks
* Persistent failure episode analysis
* Early-warning analysis
* Walk-forward comparison of alternative feature sets

A separate deterministic gateway-holdout experiment produced an AUC of approximately **0.9468** on the held-out gateway set.

This is supporting evidence for generalisation and is **not** treated as a guarantee of live performance or as the official challenge cost result.

---

## Submission Output

The final prediction pipeline generates:

```text
predictions.csv
```

The required output contains:

* **120 rows**
* **8 prediction weeks**
* **15 gateways per week**
* Ranks 1–15 for each week
* Gateway ID
* Model score
* Operational reason

The supplied `validate_submission.py` validator is run against the generated submission.

---

## How to Run

### Requirements

* Python 3.10+ recommended
* Docker Desktop (for the reproducible Docker workflow)

### Option 1 — Docker

The recommended way to run the complete pipeline is:

```powershell
docker compose run --rm lpdg-ml
```

This:

1. Builds the required environment.
2. Runs the final ML pipeline.
3. Generates `predictions.csv`.
4. Copies the result to `outputs/predictions.csv`.
5. Runs the submission validator.

A successful run should report:

```text
Rows: 120
Weeks: 8
Gateways per week: [15, 15, 15, 15, 15, 15, 15, 15]
```

and the validator should report:

```text
predictions.csv: OK
```

### Option 2 — Python

Install the dependencies:

```powershell
pip install -r requirements.txt
```

Run the prediction pipeline:

```powershell
python src/final_predict.py
```

Then validate the generated file:

```powershell
python validate_submission.py predictions.csv
```

---

## Repository Structure

```text
LPDG-x-RGM-Innovation-Challenge/
│
├── README.md
├── DECISIONS.md
├── AI-USAGE.md
├── baseline_3sigma.py
├── validate_submission.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── eda_profile.py
├── eda_deep.py
│
├── src/
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── compare_models.py
│   └── final_predict.py
│
└── outputs/
    └── .gitkeep
```

The challenge datasets are kept outside Git tracking using `.gitignore`.

---

## Visualizations

The following visualizations summarize the main operational evaluation results.

### 1. Total Operational Cost — ML vs 3-Sigma Baseline

This comparison shows the total operational cost over the 8-week temporal
validation period.

The ML ranking produced a total cost of **€129,600**, compared with
**€130,800** for the supplied 3-sigma baseline.

![Total Operational Cost: ML vs 3-Sigma Baseline](plots/ml_vs_baseline_cost.png)

### 2. Weekly Operational Cost Comparison

The weekly comparison shows how the ML ranking performed against the baseline
throughout the validation period.

ML did not outperform the baseline in every individual week. The overall
improvement comes from the result across the complete 8-week validation period.

![Weekly Operational Cost: ML vs 3-Sigma Baseline](plots/weekly_cost_comparison.png)


## Limitations

### 1. Early failure detection

The strongest behaviour observed during analysis was for persistent or already-severe failure patterns.

I do **not** claim that the current model is a reliable early-warning system for sudden failures.

A gateway with little historical evidence can still fail unexpectedly.

### 2. Physical diagnosis

The model predicts risk based on telemetry patterns.

It does not determine the actual physical cause of a gateway problem, such as:

* Hardware failure
* Connectivity problems
* Power problems
* Network configuration
* Meter-side issues
* Environmental conditions

The `reason` field is therefore an operational explanation based on recent signals, not a physical diagnosis.

### 3. Telemetry coverage

Incomplete or sparse telemetry means that there may be less evidence available for some gateways.

Telemetry coverage is included as a feature, but it does not remove this uncertainty.

Quiet or missing gateways should not automatically be interpreted as healthy.

### 4. Changing gateway population

The gateway population can change over time.

New gateway IDs, disappearing gateways and gateways that stop reporting need to be handled carefully during live evaluation.

### 5. Field-visit feedback

Field visits were not used directly as the prediction target because the decision to visit a gateway is selective.

The current model learns from meter-read outcomes instead.

A future version could incorporate verified field outcomes while explicitly accounting for intervention bias.

---

## What I Would Improve Next

If I had another two weeks, I would prioritise:

### 1. Recalibrate the decision threshold

Use verified field outcomes from recommended visits to determine whether the current >30% target remains the best operational choice.

### 2. Improve early-warning evaluation

Build a dedicated evaluation for newly emerging severe failures and measure:

* How many weeks in advance they can be detected.
* Whether they enter the top 15 before becoming severe.
* The operational cost of missing them.

### 3. Monitor gateway population changes

Add explicit checks for:

* New gateway IDs
* Gateways that stop reporting
* Unexpected telemetry coverage drops
* Changes in the number of candidate gateways

The goal would be to flag unexpected changes instead of silently producing a normal-looking ranking.

---

## Reproducibility

The project is designed so that the complete prediction and validation workflow can be executed locally using the supplied data and Docker configuration.

The final pipeline keeps data processing, feature engineering, model training, prediction and validation explicit.

The same workflow was tested locally and through Docker before submission.

---

## Documentation

* [`DECISIONS.md`](DECISIONS.md) — modelling decisions, alternatives, validation and limitations.
* [`AI-USAGE.md`](AI-USAGE.md) — how AI assistance was used during development.

---

## Final Note

The final solution intentionally favours a model that is simple enough to understand, reproduce and modify rather than one that performs better on only a single metric or split.

The important question is not only:

> **"How accurate is the model?"**

but also:

> **"Does the ranking help the field team make a better decision with limited visit capacity, and do we understand when the model might be wrong?"**

````


