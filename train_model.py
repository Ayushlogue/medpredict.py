import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("   DISEASE PREDICTOR - MODEL TRAINING")
print("=" * 50)


df = pd.read_csv("disease_symptoms.csv")


symptom_cols = [c for c in df.columns if c.startswith("Symptom")]
all_symptoms = set()
for col in symptom_cols:
    all_symptoms.update(df[col].dropna().str.strip().str.lower())
all_symptoms = sorted(all_symptoms)

print(f"\n✅ Total diseases  : {df['Disease'].nunique()}")
print(f"✅ Total symptoms  : {len(all_symptoms)}")


def encode_row(row):
    present = set()
    for col in symptom_cols:
        val = row.get(col)
        if pd.notna(val):
            present.add(str(val).strip().lower())
    return [1 if s in present else 0 for s in all_symptoms]

X = np.array([encode_row(row) for _, row in df.iterrows()])
y = df["Disease"].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


print("\n🌳 Training Decision Tree...")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt.predict(X_test))
print(f"   Decision Tree Accuracy : {dt_acc * 100:.2f}%")


print("🌲 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"   Random Forest Accuracy : {rf_acc * 100:.2f}%")


best_model = rf if rf_acc >= dt_acc else dt
best_name  = "Random Forest" if rf_acc >= dt_acc else "Decision Tree"
print(f"\n🏆 Best Model : {best_name} ({max(rf_acc, dt_acc)*100:.2f}%)")


with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("symptoms_list.pkl", "wb") as f:
    pickle.dump(all_symptoms, f)

print("\n✅ model.pkl saved")
print("✅ symptoms_list.pkl saved")
print("\n🚀 Training complete! You can now run: python app.py")
print("=" * 50)
