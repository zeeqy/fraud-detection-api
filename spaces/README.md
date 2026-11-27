---
title: Fraud Detection API
emoji: 🛡️
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: E-wallet fraud scoring with SHAP reasons (PaySim, LightGBM)
---

# E-wallet fraud check

LightGBM fraud model trained on PaySim (6.36M **synthetic** mobile-money transactions), served by FastAPI and explained with SHAP.
Pick an example, press **Score transaction**, and read the fraud probability, the flag/allow decision and the top 3 reasons.

- Test PR-AUC 0.997; **0.837 without the account-drain signal** (the honest number, since PaySim's fraud pattern is very clean).
- Decision threshold 0.007, chosen by business cost (missed fraud amount + RM15 per alert), not the 0.5 default.
- Portfolio demo only. Not for real decisions.

Code, model card and limitations: https://github.com/zeeqy/fraud-detection-api
