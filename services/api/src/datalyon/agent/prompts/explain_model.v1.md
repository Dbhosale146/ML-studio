Facts about the final model (all computed by code):
<data>{{facts}}</data>
Return:
- explanation: 3 sentences for a business user: what drives the predictions (name the top drivers) and how accurate the model is, in plain words.
- why_model: 2-3 sentences on why this algorithm won, citing the runner-up and baseline scores from the facts.

How to describe accuracy honestly:
- Lead with the primary metric (`metric`, `score`) and compare it with its `baseline`. That comparison is the headline.
- Each holdout metric comes with its naive baseline. Only call a metric good if it clearly beats that baseline. Never praise a metric that is close to or below its baseline.
- If `majority_class_share` is 70% or more, do not cite accuracy as evidence of quality: always guessing the most common class already scores about that much. If you mention accuracy at all, say this.
- Do not use words like "strong", "excellent" or "highly accurate" unless the primary metric is far above its baseline. Plain, measured wording is better than praise.
