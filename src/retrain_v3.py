import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

D = "/home/claude/data/lahman_1871-2025_csv/"
batting = pd.read_csv(D+"Batting.csv")

# Full league context per year: K%, BB%, SB rate
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

df["K_plus_proxy"] = df.K_pct / df.league_K_pct       # >1 = worse (more Ks than avg)
df["BB_plus_proxy"] = df.BB_pct / df.league_BB_pct     # >1 = better (more walks than avg)
df["SB_rate_own"] = df.SB / df.PA
df["SB_plus_proxy"] = df.SB_rate_own / df.league_SB_rate

FEATURES = ["age","PA","OPS_plus_proxy","HR_plus_proxy","BB_plus_proxy","K_plus_proxy","SB_plus_proxy"]
df2 = df.dropna(subset=FEATURES+["primary_pos"])
df2 = df2.replace([np.inf,-np.inf], np.nan).dropna(subset=FEATURES)

pos_dummies = pd.get_dummies(df2["primary_pos"], prefix="pos")
X = pd.concat([df2[FEATURES], pos_dummies], axis=1)
y = df2["outcome"]

train_mask = df2.rookie_year < 2005
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10, random_state=42, class_weight="balanced")
clf.fit(X_train_s, y_train)
pred = clf.predict(X_test_s)

print("\nAccuracy (fully normalized incl. K%/BB%/SB):", round(accuracy_score(y_test, pred), 3))
print(classification_report(y_test, pred))
importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances.head(8))
