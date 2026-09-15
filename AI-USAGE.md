# AI-USAGE.md

## How I Used AI

I used AI as a supporting tool during this project, mainly to get another
perspective while I was working through the challenge.

I did not use AI as a replacement for understanding the problem, working with
the data, running experiments, or making the final technical decisions.

The code and results in this repository were run and checked by me on the
provided challenge data.

---

## 1. Understanding the Challenge

I first read the challenge brief and worked through the requirements,
constraints, costs, data, and expected outputs myself.

After forming my own understanding, I used AI to discuss the problem and to
get another perspective on possible approaches and things I might have
overlooked.

I then compared those suggestions with the actual challenge requirements.

The final interpretation of the problem and the decisions made in the
project were my own.

---

## 2. Understanding the Data

I spent time exploring the provided datasets before building the ML solution.

I inspected the schemas, time ranges, relationships between the datasets, and
the available gateway information myself.

While working with the data, I used AI when I had questions about Python,
pandas, data processing, or possible ways to analyse a particular pattern.

I did not assume that an AI suggestion was correct. I checked the suggested
approach against the actual data and the challenge requirements before using
it.

---

## 3. EDA and Feature Engineering

AI was mainly useful to me as a second perspective during EDA and feature
engineering.

I used it to:

- understand Python and pandas errors,
- discuss possible causes of unexpected results,
- consider alternative ways of analysing patterns,
- discuss possible feature-engineering ideas,
- and help reason about implementation issues.

I then implemented and tested the relevant ideas myself.

The EDA findings documented in this repository are based on experiments I
actually ran on the challenge data.

For example, I checked telemetry behaviour, failure-rate patterns, telemetry
coverage, gateway behaviour over time, and the relationship between telemetry
features and failures.

---

## 4. Model Selection and Experimentation

I used AI to discuss possible modelling approaches and feature ideas.

However, suggestions were treated as hypotheses to test rather than as final
answers.

The final model and feature choices were made based on the experiments and
validation results I obtained.

One example was the experiment with additional trend-based features.

The trend features initially looked promising on one fixed validation setup:

Base model  -> €129,600
Trend model -> €127,800

Instead of accepting that result immediately, I tested the approach using a
walk-forward evaluation as well.

The result changed:

Base model  -> €208,300
Trend model -> €211,240

Since the trend-based version had a higher cost in the walk-forward
evaluation, I rejected it and kept the simpler 69-feature Logistic Regression
model.

This was an important part of my modelling process: I used AI suggestions to
generate ideas, but relied on experiments and validation to decide whether an
idea should actually be used.

---

## 5. Debugging and Implementation

During implementation, I used AI to help understand errors and think through
implementation problems.

This included issues related to:

- pandas and data processing,
- feature construction,
- model training,
- validation,
- Git and repository workflow,
- and making the prediction pipeline easier to run.

When AI suggested a solution, I tested it in my environment before keeping
the change.

I also checked that changes did not introduce data leakage or violate the
challenge requirements.

---

## 6. Validation and Final Decisions

AI did not determine the final model performance or validation results.

I ran the experiments myself and used the results to make the final
decisions.

In particular, I considered:

- operational cost rather than accuracy alone,
- temporal validation,
- unseen-gateway validation,
- feature coverage,
- model simplicity,
- and the risk of relying on patterns that may not generalize.

The final decisions are documented separately in `DECISIONS.md`.

---

## 7. Documentation

I used AI to help review and organize some documentation and to make some
technical explanations clearer.

The technical findings, experiment results, limitations, and final decisions
in the repository are based on the work I performed and verified.

AI was used as a writing and review aid, not as a source for inventing
results.

---

## One Thing AI Got Wrong

One useful example happened while I was exploring additional trend-based
features.

AI suggested that adding trend information could help the model capture
changes in gateway behaviour.

I implemented and tested the idea.

The trend version initially looked better on one fixed validation experiment:

Base model  -> €129,600
Trend model -> €127,800

However, I tested the idea further using walk-forward evaluation.

The result was:

Base model  -> €208,300
Trend model -> €211,240

The trend model therefore performed worse in that evaluation.

I rejected the trend features and kept the simpler model.

This reinforced an important lesson for me: an approach that looks better on
one validation result should not automatically be considered better. It needs
to be tested under more realistic evaluation conditions.

---

## Overall

AI was a supporting tool throughout this project.

I mainly used it for:

- getting another perspective on the problem,
- discussing possible approaches,
- understanding technical errors,
- debugging implementation issues,
- exploring feature ideas,
- and reviewing documentation.

I remained responsible for understanding the data, running the experiments,
checking the results, and making the final technical decisions.

The final solution reflects the approaches that I tested and found reasonable
for the challenge, rather than simply accepting AI-generated suggestions.