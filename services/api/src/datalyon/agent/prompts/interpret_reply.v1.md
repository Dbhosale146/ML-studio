The assistant asked the user: <data>{{prompt}}</data>
Options (JSON): {{options}}
Dataset columns: {{columns}}
The user typed: <data>{{text}}</data>

Map the reply:
- choice_index: index of the option the user means, or null.
- column: an exact column name if the user named one that is not an option, else null.
- number: a number the user gave (for example a horizon), else null.
- answer: if the user asked a question instead of choosing, a short helpful answer (2-3 sentences); else null.
