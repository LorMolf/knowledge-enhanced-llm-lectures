r"""
<a name="section-20"></a>

# 20. Compare what actually happened

The shared ledger contains measured retrieval results for the same held-out
questions and closed corpus. Scores are calculated by the same evaluator;
partial relevance judgments and the small teaching corpus limit interpretation.
Let's display the results before asking which representation, training choice or
retrieval mechanism helped. Which differences are meaningful for your use case?
"""

import pandas as pd

# 1. turn the accumulated evaluation rows into one table so runs sit side by side
comparison = pd.DataFrame(evaluation_rows)

# 2. show the ledger: every row is one measured configuration
display(comparison)
