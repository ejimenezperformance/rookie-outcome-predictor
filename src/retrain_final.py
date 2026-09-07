import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

df = pd.read_csv("rookie_labeled_normalized.csv")

# CRITICAL FIX: exclude rookies who debuted before 1933 (All-Star game didn't exist yet,
# so the "Star" label is structurally near-impossible for them, not a reflection of talent)
before = len(df)
df = df[df.rookie_year >= 1933].copy()
print(f"Filtered from {before} to {len(df)} rookies (removed pre-1933 debuts)")
print(df.outcome.value_counts(normalize=True).round(3))

FEATURES = ["age","PA","OPS_plus_proxy","HR_plus_proxy","BB_pct","K_pct","SB"]
df = df.dropna(subset=FEATURES+["primary_pos"])
# also exclude pitchers from the hitter model (this project is hitters-only, per README)
df = df[df.primary_pos != "G_p"].copy()
print(f"After removing pitcher-primary rows: {len(df)}")

pos_dummies = pd.get_dummies(df["primary_pos"], prefix="pos")
X = pd.concat([df[FEATURES], pos_dummies], axis=1)
y = df["outcome"]

train_mask = df.rookie_year < 2005
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10, random_state=42, class_weight="balanced")
clf.fit(X_train_s, y_train)
pred = clf.predict(X_test_s)

print("\nFINAL Accuracy:", round(accuracy_score(y_test, pred), 3))
print(classification_report(y_test, pred))
print("\nConfusion matrix:")
labels = ["Bust","Regular","Star"]
cm = confusion_matrix(y_test, pred, labels=labels)
print(pd.DataFrame(cm, index=[f"actual_{l}" for l in labels], columns=[f"pred_{l}" for l in labels]))

importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature importances:")
print(importances.head(8))

df["_test"] = ~train_mask
df.to_csv("rookie_final.csv", index=False)
import joblib
joblib.dump({"model":clf,"scaler":scaler,"features":X.columns.tolist()}, "rookie_model_FINAL.joblib")
