import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import joblib

NAVY = "#0B1B33"
GOLD = "#D4A53A"

D = "/home/claude/data/lahman_1871-2025_csv/"
batting = pd.read_csv(D+"Batting.csv")
season = batting.groupby(["playerID","yearID"])[["AB","BB","HBP","SF","SO","SB"]].sum().reset_index()
season["PA"] = season.AB+season.BB+season.HBP+season.SF
season = season[season.AB>=50].copy()
season["K_pct"] = season.SO/season.PA
season["BB_pct"] = season.BB/season.PA
season["SB_rate"] = season.SB/season.PA
league = season.groupby("yearID")[["K_pct","BB_pct","SB_rate"]].mean()
league.columns = ["league_K_pct","league_BB_pct","league_SB_rate"]

df = pd.read_csv("rookie_final.csv")
df = df.merge(league, left_on="rookie_year", right_index=True, how="left")
df["K_plus_proxy"] = df.K_pct/df.league_K_pct
df["BB_plus_proxy"] = df.BB_pct/df.league_BB_pct
df["SB_rate_own"] = df.SB/df.PA
df["SB_plus_proxy"] = df.SB_rate_own/df.league_SB_rate
df = df.replace([np.inf,-np.inf], np.nan)

bundle = joblib.load("rookie_model_TUNED.joblib")
clf, scaler, feat_cols = bundle["model"], bundle["scaler"], bundle["features"]

FEATURES = ["age","PA","OPS_plus_proxy","HR_plus_proxy","BB_plus_proxy","K_plus_proxy","SB_plus_proxy"]
df2 = df.dropna(subset=FEATURES+["primary_pos"])
pos_dummies = pd.get_dummies(df2["primary_pos"], prefix="pos")
X_full = pd.concat([df2[FEATURES], pos_dummies], axis=1).reindex(columns=feat_cols, fill_value=0)
test_mask = df2.rookie_year >= 2005
X_test_s = scaler.transform(X_full[test_mask])
y_test = df2.outcome[test_mask]
pred = clf.predict(X_test_s)

labels = ["Bust","Regular","Star"]
cm = confusion_matrix(y_test, pred, labels=labels)
cm_pct = cm/cm.sum(axis=1,keepdims=True)*100
fig, ax = plt.subplots(figsize=(6,5))
ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
ax.set_xticks(range(3)); ax.set_xticklabels(labels)
ax.set_yticks(range(3)); ax.set_yticklabels(labels)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Rookie Outcome Predictor — FINAL (tuned)\nTest: 2005-2020 debuts | 57.3% accuracy", color=NAVY, fontweight="bold", fontsize=11)
for i in range(3):
    for j in range(3):
        ax.text(j,i,f"{cm_pct[i,j]:.0f}%\n(n={cm[i,j]})", ha="center", va="center",
                 color="white" if cm_pct[i,j]>50 else NAVY, fontsize=10)
plt.tight_layout()
plt.savefig("confusion_matrix_TUNED.png", dpi=150)

importances = pd.Series(clf.feature_importances_, index=feat_cols).sort_values(ascending=False).head(8)
fig, ax = plt.subplots(figsize=(8,5))
ax.barh(importances.index[::-1], importances.values[::-1], color=NAVY)
ax.set_title("Top Feature Importances — Final Tuned Model", color=NAVY, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("feature_importance_TUNED.png", dpi=150)
print("Charts saved.")
