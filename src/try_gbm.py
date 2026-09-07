import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

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

FEATURES = ["age","PA","OPS_plus_proxy","HR_plus_proxy","BB_plus_proxy","K_plus_proxy","SB_plus_proxy"]
df2 = df.dropna(subset=FEATURES+["primary_pos"])
pos_dummies = pd.get_dummies(df2["primary_pos"], prefix="pos")
X = pd.concat([df2[FEATURES], pos_dummies], axis=1)
y = df2["outcome"]

train_mask = df2.rookie_year < 2005
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

# Internal validation split WITHIN training data only, time-based (not random)
# to tune GBM without touching the real test set
train_sub = df2[train_mask]
inner_train_mask = train_sub.rookie_year < 1995
X_inner_train, X_inner_val = X_train[inner_train_mask.values], X_train[~inner_train_mask.values]
y_inner_train, y_inner_val = y_train[inner_train_mask.values], y_train[~inner_train_mask.values]

scaler = StandardScaler()
X_inner_train_s = scaler.fit_transform(X_inner_train)
X_inner_val_s = scaler.transform(X_inner_val)

print("Tuning GBM on inner time-split (train<1995 / val 1995-2004), test set untouched:\n")
best_acc, best_params = -1, None
for lr in [0.03, 0.05, 0.1]:
    for depth in [2, 3, 4]:
        for n_est in [100, 200, 300]:
            gbm = GradientBoostingClassifier(learning_rate=lr, max_depth=depth, n_estimators=n_est, random_state=42)
            gbm.fit(X_inner_train_s, y_inner_train)
            acc = accuracy_score(y_inner_val, gbm.predict(X_inner_val_s))
            if acc > best_acc:
                best_acc, best_params = acc, (lr, depth, n_est)

print(f"Best GBM params (inner validation): lr={best_params[0]}, max_depth={best_params[1]}, n_estimators={best_params[2]} (inner acc={best_acc:.3f})")

# Refit on FULL training set with best params, evaluate ONCE on real test set
scaler_full = StandardScaler()
X_train_s = scaler_full.fit_transform(X_train)
X_test_s = scaler_full.transform(X_test)
gbm_final = GradientBoostingClassifier(learning_rate=best_params[0], max_depth=best_params[1], n_estimators=best_params[2], random_state=42)
gbm_final.fit(X_train_s, y_train)
pred_gbm = gbm_final.predict(X_test_s)
print(f"\nGBM FINAL test accuracy: {accuracy_score(y_test, pred_gbm):.3f}")
print(classification_report(y_test, pred_gbm))

# Compare directly against our tuned Random Forest for reference
import joblib
bundle = joblib.load("rookie_model_TUNED.joblib")
rf_clf, rf_scaler, rf_feats = bundle["model"], bundle["scaler"], bundle["features"]
X_test_rf = X_test.reindex(columns=rf_feats, fill_value=0)
X_test_rf_s = rf_scaler.transform(X_test_rf)
pred_rf = rf_clf.predict(X_test_rf_s)
print(f"\n(Reference) Tuned Random Forest test accuracy: {accuracy_score(y_test, pred_rf):.3f}")
