# AI Usage

AI was used as a supporting development tool throughout this project. I used it
mainly as a technical discussion partner while understanding the challenge,
exploring approaches, debugging implementation issues, interpreting results,
and improving the project documentation.

I also used AI to generate or suggest parts of code when needed, but I tested
the code locally, checked the outputs against the challenge requirements, and
made changes based on the actual behaviour of the implementation.

## Examples of How I Used AI

### 1. Understanding the challenge and evaluation criteria

At the beginning, I used AI to break down the challenge requirements and
understand what the evaluation was actually measuring.

For example, I discussed:

- How the 15-gateway weekly selection should work.
- Why the operational cost matters more than a conventional accuracy metric.
- How the €380 wasted-visit cost and €600 per-week missed-fault cost affect
  the ranking problem.
- What temporal validation and unseen-gateway validation mean for this
  particular dataset.
- What kinds of data leakage could occur when using telemetry from the target
  week.

This helped me form a clearer understanding of the problem before deciding
which experiments to run.

### 2. Feature engineering discussions

I used AI to discuss possible telemetry features and different ways of
aggregating the hourly gateway data.

Examples included discussing:

- 7-day, 14-day and 28-day historical windows.
- Offline duration and disconnection behaviour.
- Reboots and reboot duration.
- Online duration.
- RSSI-related features.
- Telemetry coverage as a feature.
- Whether recent trends should be added to the base feature set.

I then implemented and tested these ideas on the actual dataset rather than
assuming that every suggested feature would improve the solution.

### 3. Model selection

I used AI to compare possible approaches for the ML track, including simpler
models and more complex alternatives.

The final model is Logistic Regression because it provided a good balance
between predictive performance, operational cost, interpretability and
reproducibility for this problem.

AI was useful in discussing the trade-offs, but the final choice was based on
the experiments I ran and the validation results.

### 4. Debugging and implementation

AI was also used while implementing the Python pipeline.

For example, I used it to help debug issues involving:

- Pandas data processing and merging.
- Timestamp and timezone handling.
- Gateway ID normalization.
- Building historical feature windows without using future information.
- Training and scoring the model.
- Deterministic ranking of gateways.
- Generating the required `predictions.csv`.
- Docker execution and validation.

When an issue was suggested or a code change was generated, I ran it locally
and checked whether the resulting output was actually correct.

### 5. Leakage and validation checks

One important use of AI was to challenge my assumptions about whether the
model evaluation was valid.

I discussed questions such as:

- Could target-week telemetry accidentally enter the features?
- Is using field-visit outcomes as labels appropriate?
- Does a random row split create gateway leakage?
- How should unseen gateways be evaluated?
- How should future weeks be handled?
- Does a lower AUC necessarily mean a higher operational cost, or vice versa?

These discussions led me to perform additional experiments and audits instead
of relying on a single train/test result.

### 6. Comparing ML with the supplied baseline

I used AI to help structure the cost-based comparison between my ML ranking and
the supplied 3-sigma baseline.

The important part was understanding that the objective is not simply to
maximize classification accuracy. I therefore compared the two approaches
using the challenge's operational cost definition over the complete temporal
validation period.

The final validation result was:

- ML: €129,600
- 3-sigma baseline: €130,800
- Difference: €1,200 lower cost for ML

I used the actual experiment output as the basis for this conclusion.

### 7. Investigating model behaviour

After obtaining the initial results, I used AI to suggest questions that could
help me understand where the model was succeeding or failing.

This led to additional analysis of:

- Persistent severe gateways.
- Fault episodes.
- Early detection behaviour.
- Unseen gateways.
- Different time periods.
- Feature coverage.
- Alternative ranking transformations.
- Trend features.

Some experiments improved one metric but did not improve the final operational
cost. I therefore did not automatically include every experiment in the final
pipeline.

For example, trend features improved some fixed-split metrics, but the
walk-forward cost comparison did not improve over the base model. This was one
reason I kept the simpler 69-feature base model as the final candidate.

### 8. Documentation and presentation

I also used AI while writing and organizing:

- `README.md`
- `DECISIONS.md`
- `AI-USAGE.md`
- Explanations of experiments and limitations.
- Visualization descriptions.
- Docker usage instructions.

AI helped me structure the information and improve clarity, while I checked
the documentation against the actual code and experiment results.

## What I Learned From Using AI

Using AI was most useful when I treated it as something I could question and
discuss ideas with rather than as a replacement for running experiments.

A suggestion could look reasonable but still perform poorly on the actual
dataset. In several cases, I had to test an idea, look at the resulting
metrics or operational cost, and then decide whether to keep or reject it.

This was particularly important for this challenge because the best-looking
ML metric was not always the same as the best operational decision.

Overall, AI was part of my development workflow for learning, coding,
debugging, experimentation, validation and documentation. The dataset,
experiments, outputs and final decisions were all checked as part of my own
development process.