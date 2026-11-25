Goal: <data>{{goal}}</data>
Target column (never use it): {{target}}
Numeric input columns you may combine: {{columns}}
Dataset summary:
<data>
{{summary}}
</data>

Suggest up to 5 derived features a senior data scientist would add for this goal, using only these operations:
ratio (a / b), diff (a - b), sum (a + b), product (a * b), log1p (log(1 + a), single column).
Each feature: name (snake_case), op, a, b (null for log1p), why (one short sentence).
They will only be kept if they improve cross-validated accuracy.
