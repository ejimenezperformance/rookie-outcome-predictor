import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

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

FEATURES = ["age","PA","OPS_plus_proxy","HR_plus_proxy","BB_plus_proxy","K_plus_proxy","SB_plus_proxy"]
df2 = df.replace([np.inf,-np.inf], np.nan).dropna(subset=FEATURES+["primary_pos"])
pos_dummies = pd.get_dummies(df2["primary_pos"], prefix="pos")
X = pd.concat([df2[FEATURES], pos_dummies], axis=1)
y = df2["outcome"]

train_mask = df2.rookie_year < 2005
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("Grid search using OOB score on TRAINING data only (test set untouched):\n")
best_oob = -1
best_params = None
results = []
for max_depth in [4,6,8,10,12,None]:
    for min_leaf in [5,10,20,30]:
        clf = RandomForestClassifier(n_estimators=500, max_depth=max_depth, min_samples_leaf=min_leaf,
                                      random_state=42, class_weight="balanced", oob_score=True, n_jobs=-1)
        clf.fit(X_train_s, y_train)
        oob = clf.oob_score_
        results.append((max_depth, min_leaf, oob))
        if oob > best_oob:
            best_oob = oob
            best_params = (max_depth, min_leaf)

results_df = pd.DataFrame(results, columns=["max_depth","min_samples_leaf","oob_score"]).sort_values("oob_score", ascending=False)
print(results_df.head(10).to_string(index=False))
print(f"\nBest params: max_depth={best_params[0]}, min_samples_leaf={best_params[1]} (OOB={best_oob:.3f})")

# Fit final model with best params, evaluate ONCE on true test set
clf_final = RandomForestClassifier(n_estimators=500, max_depth=best_params[0], min_samples_leaf=best_params[1],
                                     random_state=42, class_weight="balanced", n_jobs=-1)
clf_final.fit(X_train_s, y_train)
pred = clf_final.predict(X_test_s)
print(f"\nFINAL TEST accuracy with tuned hyperparameters: {accuracy_score(y_test, pred):.3f}")
print(classification_report(y_test, pred))

import joblib
joblib.dump({"model":clf_final,"scaler":scaler,"features":X.columns.tolist(),"best_params":best_params},
            "rookie_model_TUNED.joblib")
