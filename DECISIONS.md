# DECISIONS.md

This document records the five main decisions I made while building my
solution for the LPDG x RGM Innovation Challenge.

For each decision, I describe what I chose, what else I could have done, and
why I selected the final approach.

I made these decisions after first understanding the challenge requirements,
then spending time understanding the provided data, performing EDA, and
running experiments on the actual dataset.

---

# Decision 1 — Choose Machine Learning for Part 2

## What I chose

For Part 2, I selected the **Machine Learning** area.

## Why I chose it

After reading the challenge brief, I understood that the field team has
limited capacity and needs to decide which gateways should be visited first.

Only 15 gateways can be visited in a week, so I wanted to build a model that
could combine multiple gateway signals and produce a risk score that could be
used for ranking.

During my initial analysis, I found relationships between gateway failure and
several telemetry signals, including:

- offline duration,
- disconnection activity,
- reboot activity,
- online duration,
- and telemetry coverage.

This made Machine Learning a suitable area for me to explore because I could
use these signals together rather than depending on one manually selected
rule.

## What else I could have done

I could have chosen the **Data Science** area and focused more on direct
analysis, thresholds, operational patterns, and decision rules without
building an ML model.

## Why I did not choose it

I wanted to investigate whether a predictive model could combine the telemetry
signals into a useful risk score and improve the gateway ranking compared with
the supplied 3-sigma baseline.

Machine Learning also gave me an opportunity to test the solution using
future weeks and gateways that were not used for training.

---

# Decision 2 — Define Severe Failure as >30% Failure Rate

## What I chose

I derived the weekly failure rate from the meter-read data:

```text
failure_rate = 1 - meters_read / meters_expected
```

I then defined the ML target as:

```text
failure_rate > 30%
```

## What else I could have done

I could have used a different threshold to define a severe failure.

I tested:

- >10%
- >20%
- >30%

## What I found

The validation costs were:

| Target threshold | Validation cost |
|---|---:|
| >10% | €155,400 |
| >20% | €142,800 |
| **>30%** | **€129,600** |

## Why I chose it

The >30% threshold produced the lowest validation cost among the thresholds I
tested.

Therefore, I selected it for the final ML experiments.

This is a modelling decision based on the data and experiments in this
project. I am not claiming that 30% is a universal definition of gateway
failure.

---

# Decision 3 — Use 7/14/28-Day Telemetry Features and Keep Coverage

## What I chose

I created historical telemetry features using:

- 7-day windows,
- 14-day windows,
- 28-day windows.

The features include telemetry signals such as:

- offline duration,
- disconnection count,
- reboot count,
- reboot duration,
- online duration,
- load,
- uptime,
- signal-quality information,
- importance indicators,
- and telemetry coverage.

The final feature set contains **69 model features**.

## Why I chose these features

I spent several days understanding the data and performing EDA before deciding
which signals were useful.

The EDA showed relationships between failure rate and several telemetry
signals.

For example:

- higher offline duration was associated with higher failure rate,
- higher disconnection activity was associated with higher failure rate,
- reboot activity also showed a relationship with failure,
- online duration showed an inverse relationship with failure,
- telemetry coverage provided useful information about gateway behaviour.

I used multiple time windows because I wanted the model to capture both recent
and persistent behaviour.

```text
7 days   → recent behaviour
14 days  → short-term behaviour
28 days  → longer-term behaviour
```

## Coverage experiment

I also tested whether telemetry coverage should be removed.

The results were:

```text
69 features with coverage    → €129,600
66 features without coverage → €132,600
```

Removing coverage resulted in a higher validation cost, so I kept it.

## What else I could have done

I could have:

1. Used only a recent short window.
2. Removed telemetry coverage.
3. Added additional trend-based features.

I also tested trend-based features.

The trend version improved one fixed validation experiment:

```text
Base model  → €129,600
Trend model → €127,800
```

However, the walk-forward evaluation gave:

```text
Base model  → €208,300
Trend model → €211,240
```

## Why I did not add the trend features

The trend features did not consistently improve the operational cost across
the validation approaches I tested.

Since they also added complexity, I decided to keep the simpler 69-feature
set.

This decision was based on the actual experimental results rather than
choosing the more complex feature set just because it performed better on one
validation split.

---

# Decision 4 — Use Logistic Regression as the Final Model

## What I chose

I selected **Logistic Regression** as the final ML model.

The implemented pipeline uses:

- median imputation,
- standard scaling,
- balanced class weights,
- `max_iter=2000`,
- `random_state=42`.

The model produces a risk score for each gateway.

## Why I chose it

I wanted a model that was:

- simple,
- fast,
- reproducible,
- understandable,
- easy to validate,
- and easy to modify.

The challenge also includes a live evaluation where I may need to explain and
change my own code.

For that reason, I preferred a model that I could understand completely
rather than adding complexity without clear evidence of a better operational
result.

## What else I could have done

I could have used a more complex ML model or continued adding more feature
engineering.

I also experimented with trend-based features.

Another possibility was to transform the model scores before ranking them.

## Why I did not choose those alternatives

The trend experiment did not consistently improve the operational cost.

Also, a monotonic transformation of the predicted scores would not change
their ordering, so it would not improve the gateway ranking itself.

Therefore, I kept the simpler Logistic Regression approach.

The final ranking sorts gateways by predicted risk in descending order and
uses `gateway_id` as a deterministic tie-breaker.

---

# Decision 5 — Evaluate Using Operational Cost and Stronger Validation

## What I chose

I evaluated the solution primarily using the **operational cost defined by
the challenge**, while also using temporal and gateway-holdout checks to
understand model behaviour.

The field team has a hard limit of 15 visits per week, so the final decision
is based on the ranking of those 15 gateways.

## Why I chose operational cost

The challenge has different costs for different mistakes:

- €380 for a wasted field visit.
- €600 for leaving a faulty gateway unattended for one week.
- The €600 cost continues for additional weeks until the gateway is visited.

Therefore, accuracy, precision, recall, F1, or AUC alone do not represent the
actual operational objective.

I wanted to know whether the ranking produced a better field-visit decision,
not simply whether the model classified gateways correctly.

## ML vs baseline result

For the historical validation period I tested:

```text
ML model      → €129,600
3-sigma       → €130,800
```

The ML approach therefore had:

```text
€1,200 lower cost
```

than the supplied baseline on that validation period.

This is a historical validation result and is not a guarantee of future
performance.

## Temporal validation

I used time-based validation because gateway telemetry contains repeated
observations from the same gateways over time.

The main validation setup used:

```text
Training  → up to 2025-12-01
Validation → 2025-12-08 to 2026-01-26
```

A random row-level split could put observations from the same gateway into
both training and validation, which could make the result look more
optimistic.

Therefore, I preferred a temporal evaluation for the main cost comparison.

## Unseen-gateway check

I also performed a separate deterministic gateway-holdout experiment to
check how the model behaved on gateways that were not used during training.

That experiment produced an AUC of approximately:

```text
0.9468
```

on the held-out gateway set.

This was a supporting generalization check, **not the official challenge cost
evaluation and not a guarantee of live performance**.

## What else I could have done

I could have relied mainly on:

- accuracy,
- F1 score,
- AUC,
- or a random row-level train/test split.

These would have been simpler to evaluate.

## Why I did not use them as the main decision

The actual challenge is about selecting 15 gateways under a specific cost
structure.

Also, random row-level splitting can leak gateway-specific patterns across
training and validation because the same gateways appear repeatedly over
time.

Therefore, I used operational cost as the main practical measure and used
temporal and unseen-gateway checks to understand whether the model's behaviour
was likely to generalize.

---

# Final Summary of the Five Decisions

The final solution came from these five decisions:

```text
1. Part 2 area
   Machine Learning
          ↓
2. Target
   Failure rate > 30%
          ↓
3. Features
   7/14/28-day telemetry + coverage
          ↓
4. Model
   Logistic Regression
          ↓
5. Evaluation
   Operational cost + temporal/unseen-gateway checks
          ↓
   Deterministic top-15 gateway ranking
```

The final approach was kept deliberately simple because the experiments did
not provide enough evidence to justify additional complexity.

The main principle behind the decisions was:

> **Choose an approach based on evidence from the actual data and experiments,
> while keeping the solution simple enough to understand, explain, validate,
> and modify.**