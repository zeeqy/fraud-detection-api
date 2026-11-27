# DEMO: Fraud Detection API (60 to 90 s)

**Hook (on screen: the Streamlit demo, a transaction flagged in red):**
"Before an e-wallet transfer goes through, a fraud team has about a tenth of a second to decide: let it pass or hold it. This API makes that call and says why."

## Shots

| # | Show on screen | Say |
|---|---|---|
| 1 (10 s) | README top: problem table and the PaySim line | "I trained on PaySim: 6.36 million synthetic mobile-money transactions, only 0.13% fraud. So accuracy is useless: calling everything legit scores 99.7%." |
| 2 (15 s) | `notebooks/01_eda.ipynb`, leakage cell | "The trap in this dataset is the post-transaction balances. 90% of legit rows don't even add up, so they basically leak the label, and a real pre-authorisation system can't see them. I dropped them." |
| 3 (15 s) | Streamlit demo: pick "Suspicious: night transfer emptying the account", press Score | "A RM250,000 transfer at 3 a.m. that empties the account: 100% fraud probability, flagged for review, and the top three SHAP reasons in plain English. Normal cash-out: 0.001%, allowed." |
| 4 (15 s) | `/docs` page, then terminal `curl` to `/predict` | "It's a real FastAPI service: single and batch scoring, health and model-info endpoints, Pydantic validation that returns 422 on bad input. About 75 to 125 ms per call with explanations, 26 tests, Docker, CI." |
| 5 (15 s) | README results table, ablation row highlighted | "Test PR-AUC is 0.997, which is too good. So I removed the account-drain signal and retrained: 0.837. That's the honest number, and it's where LightGBM beats logistic regression, 0.84 against 0.29." |
| 6 (10 s) | `reports/figures/cost_curve.png` | "The threshold isn't 0.5. It's set by cost: missed fraud amount plus RM15 per alert. For the weaker model that cuts test losses from RM75 million to RM5.3 million, at the price of 30 times more alerts." |

**Closing (on screen: GitHub repo + Space link):**
"Leakage found, honest ablation, business-cost threshold, explanations on every decision. Code and model card are on GitHub, and you can try it live on Hugging Face."
