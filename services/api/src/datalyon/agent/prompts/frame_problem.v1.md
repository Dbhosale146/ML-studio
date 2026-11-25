Frame the user's goal as a machine-learning task.

Goal: <data>{{goal}}</data>

Dataset summary (column names, roles and statistics; personal data is masked):
<data>
{{summary}}
</data>

A rule-based detector proposed (use it as a starting point, override it only with a clear reason):
<data>{{heuristic}}</data>

Rules:
- problem_type is one of binary, multiclass, regression, forecasting.
- forecasting only when there is a date column and the goal is about future values of a numeric column over time; then set time_column, and series_column if the data holds several series (for example one per store).
- target, time_column and series_column must be exact column names from the summary, or null.
- positive_class (binary only) is the target value that means the event of interest (for example "Yes" for churned).
- horizon is the number of periods to forecast if the goal states it, else null.
- confidence is 0..1. Use below 0.6 when the goal does not make the target clear, and then write a short clarifying_question.
- signals: 3-4 short evidence bullets that cite column names. alternatives: other framings you considered and why you rejected them.
- title: a short model name of 2-5 words.
- leakage_suspects: columns that are probably only filled in after the outcome is known (for example a cancellation reason for churn).
