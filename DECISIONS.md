# Decisions
## Overview
This project ranks 15 gateways each week for field visits using historical telemetry
and meter-read outcomes.
The goal is not simply to maximize classification accuracy. The ranking is intended
to support an operational decision: which gateways should receive the limited field
capacity first.
The main design choices below describe what I chose, what I considered instead,
and why I rejected the alternatives.
---
# Decision 1 — Choose Machine Learning for Part 2
## Choice
I selected **Machine Learning** as my Part 2 area.
The problem naturally supports learning a relationship between historical gateway
telemetry and subsequent meter-read failure. A trained model can combine multiple
signals such as offline duration, disconnections, reboots, uptime, signal quality,
load and telemetry coverage instead of relying on a single manually chosen rule.
The selected area also matches the operational problem well: the system already
produces a ranked list, and the Machine Learning live-session change is to improve
the model on a month it has not seen.
## Alternative considered
I considered choosing **Data Science** instead.
That would allow the solution to focus more directly on threshold selection,
cost trade-offs and analysis of the €380 visit cost versus the €600 weekly cost of
leaving a broken gateway unattended.
## Why I did not choose it
The analysis showed that a predictive model could provide a useful improvement
over the supplied 3-sigma ranking while still allowing the operational costs to
remain the final evaluation criterion.
Machine Learning also gives me a concrete model that can be tested on both
unseen gateways and later weeks.
---
# Decision 2 — Define the ML target as severe meter-read failure (>30%)
## Choice
I defined a gateway-week as a positive ML target when its meter-read failure
rate is **greater than 30%**.
The target is based on `meter_read_success.csv`:
    failure_rate = 1 - meters_read / meters_expected
The model therefore learns to distinguish gateway-weeks with severe meter-read
failure from the remaining gateway-weeks.
The 30% threshold was chosen because the operational objective is to identify
gateways where a field visit is most likely to prevent meaningful continued
failure, rather than attempting to predict every small fluctuation in
meter-read success.
## Alternative considered
I evaluated lower and higher thresholds, including:
- >10%
- >20%
- >30%
For the tested validation period, the resulting operational costs were:
- >10% threshold: €155,400
- >20% threshold: €142,800
- >30% threshold: €129,600
## Why I did not choose the alternatives
The >30% definition produced the lowest operational cost among the tested
thresholds.
I therefore preferred a definition that focuses the model on the more serious
failure cases instead of spending limited field capacity on relatively small
failure rates.
This is an operational choice, not a claim that 30% is a universal definition
of a faulty gateway. If future field outcomes show that a different threshold
creates better decisions, the threshold should be recalibrated.
---
# Decision 3 — Use 7/14/28-day telemetry windows and retain telemetry coverage
## Choice
I created rolling historical telemetry features using **7-day, 14-day and
28-day windows** before each prediction week.
The feature set summarizes signals including:
- offline duration
- disconnection count
- reboot count
- reboot duration
- online duration
- load
- uptime
- signal quality
- importance indicators
- telemetry coverage
The final model uses **69 features**.
For each prediction week, features are constructed using information available
before that week. The target week's telemetry is not used to construct its
features.
## Alternative considered
I considered:
1. using only a short recent window;
2. removing telemetry coverage;
3. adding trend features describing changes between windows.
The coverage ablation showed:
- 69 features with coverage: AUC ≈ 0.9415, operational cost €129,600
- 66 features without coverage: AUC ≈ 0.9394, operational cost €132,600
I also tested additional trend features. They improved some fixed-split metrics,
but the improvement did not consistently survive temporal evaluation. In the
walk-forward evaluation, the trend model cost more than the base model.
## Why I did not choose the alternatives
A single short window could miss persistent conditions that develop over a
longer period.
Removing coverage discarded information about how much telemetry evidence was
available and slightly worsened both model discrimination and operational cost.
I rejected the trend features because I preferred the simpler 69-feature model
whose advantage was more consistent across the validation checks I performed.
---
# Decision 4 — Use Logistic Regression as the final model
## Choice
I selected **Logistic Regression** with:
- median imputation
- standard scaling
- balanced class weights
- `max_iter=2000`
- deterministic `random_state=42`
The model is trained on the historical labelled gateway-week dataset and then
used to score all gateways for each prediction week.
The final prediction ranking sorts scores in descending order and uses
`gateway_id` as the deterministic secondary sort key.
## Alternative considered
I considered more complex modelling approaches and additional feature
engineering, particularly trend-based features.
I also considered applying monotonic transformations to the model score for
ranking.
## Why I did not choose them
The challenge evaluates the quality of the ranking and its operational cost,
not whether the output probabilities are perfectly calibrated.
Monotonic transformations do not change the ranking, so they do not provide an
operational benefit.
The additional trend model improved some fixed-split metrics but performed worse
in the walk-forward cost evaluation:
- Base model: €208,300
- Trend model: €211,240
The simpler Logistic Regression model was therefore retained.
It also makes the model easier to inspect and modify during the live Machine
Learning session.
---
# Decision 5 — Evaluate using operational cost, temporal validation and unseen gateways
## Choice
I evaluated the model using the challenge's operational cost rather than relying
only on accuracy, precision, recall or AUC.
The ranking has a hard capacity of 15 visits per week.
For the historical validation period, I compared the ML ranking against the
supplied 3-sigma baseline.
Using an 8-week temporal holdout:
- ML total cost: **€129,600**
- Baseline total cost: **€130,800**
- Difference: **€1,200 lower cost for ML**
- Relative improvement: approximately **0.92%**
I also performed separate checks for:
- gateways not seen during model training;
- future weeks;
- different time regimes;
- persistent failure episodes.
For unseen gateways, a separate deterministic gateway-holdout experiment
produced an AUC of approximately 0.9468. This was a supporting generalization
check, not the official challenge cost evaluation, and is not treated as a
guarantee of live performance.
## Alternative considered
A random row-level train/test split would have been easier and would have
produced a convenient single metric.
I also considered evaluating only on gateways that the model had already seen.
## Why I did not choose that as the main validation
Gateway telemetry is repeated over time, so a random row split can place the same
gateway on both sides of the split. That can make the evaluation optimistic.
The challenge expects evidence on both:
1. gateways the model has never seen; and
2. weeks that occur after the training period.
Therefore, temporal and gateway-disjoint evaluation provide a more realistic test
of whether the model can generalize.
I still report model metrics such as AUC where useful, but the final operational
decision is based on the cost of the 15-gateway ranking.
---
# What it cannot do
The model has several important limitations.
## 1. It does not reliably prove early failure
The strongest behaviour observed during analysis was on gateways with persistent
or already-severe failure patterns.
In the analysed episode period, most severe cases were persistent rather than
new failures appearing for the first time.
Therefore, I do **not** claim that the model is a reliable early-warning system.
A gateway with little historical evidence can still fail suddenly.
## 2. It does not know the physical cause of a failure
The model identifies patterns associated with higher failure risk.
It does not establish whether the actual cause is:
- hardware failure;
- connectivity;
- power;
- network configuration;
- meter-side issues;
- environmental conditions;
- or another physical cause.
The `reason` field is therefore an operational explanation based on recent
signals, not a diagnosis.
## 3. It can be affected by telemetry coverage
A gateway with incomplete or unusually sparse telemetry can have less evidence
available for prediction.
Coverage is included as a feature, but this does not eliminate uncertainty.
The system should make missing or quiet gateways visible rather than treating
missing observations as proof that a gateway is healthy.
## 4. It assumes the input schema remains compatible
The current pipeline expects the challenge's telemetry, gateway, and
meter-read-success structure.
The live evaluation is expected to use the same schema, but the gateway
population may change.
A gateway can disappear from reporting, and new gateway identifiers can appear.
The pipeline therefore needs to be checked whenever the gateway population
changes.
## 5. It does not learn from field visits automatically
Field visits are not used as the prediction target because the decision to visit
was itself selective.
The current model learns from meter-read failure outcomes rather than treating
"visited" or an engineer's review category as ground truth.
A future system should incorporate verified field outcomes carefully so that
intervention bias is handled explicitly.
---
# What another two weeks would fix
If I had two additional weeks, I would prioritize the following work.
## 1. Recalibrate the decision threshold using new field outcomes
First, I would collect the outcomes of the recommended visits and evaluate
whether the >30% severe-failure definition still produces the best operational
decisions.
I would test the threshold against both missed failures and unnecessary visits,
with the costs made explicit.
This would turn the current modelling threshold into a continuously validated
operational policy.
## 2. Improve early-warning validation
The current evidence is stronger for persistent severe gateways than for newly
emerging failures.
I would construct a dedicated early-warning evaluation:
- identify gateways that become severely faulty for the first time;
- exclude information from the failure week itself;
- measure how many weeks in advance the model ranks them in the top 15;
- quantify the cost of missing them.
This would directly test whether the model is detecting deterioration early
rather than simply recognizing an already-bad gateway.
## 3. Add monitoring checks for gateway-population changes
I would add explicit checks for:
- new gateway IDs;
- gateways that have stopped reporting;
- unexpected telemetry coverage drops;
- changes in the number of candidate gateways.
The goal would be to fail loudly or flag the situation rather than silently
producing a ranking that looks normal.
---
# Final principle
The final system intentionally favors a model that is simple enough to understand,
reproduce and change over one that wins on a single metric or split.
The most important evidence is therefore not just the model score. It is whether
the ranking produces a useful operational decision, whether the result survives
unseen-gateway and future-week testing, and whether the limitations are clear
enough for an operations team to know when not to trust it.
