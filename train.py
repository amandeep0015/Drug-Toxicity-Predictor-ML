import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt

# 1. Load data
X = np.load('X.npy')
y = np.load('y.npy')

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Model
print("Training XGBoost Model and analyzing features...")
model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)

# 4. Save the model
joblib.dump(model, 'tox_model.pkl')
print(f"Model Accuracy: {model.score(X_test, y_test)*100:.2f}%")

# 5. NEW: Generate Feature Importance Chart
plt.figure(figsize=(10,6))
importances = model.feature_importances_
indices = np.argsort(importances)[-15:] # Top 15 most important molecular bits
plt.title("Key Molecular Features Linked to Toxicity")
plt.barh(range(len(indices)), importances[indices], color='crimson', align='center')
plt.yticks(range(len(indices)), [f"Molecular Bit {i}" for i in indices])
plt.xlabel("Relative Importance Score")
plt.tight_layout()

# SAVE THE FILE
plt.savefig('feature_importance.png')
print("Success! 'feature_importance.png' has been saved to your folder.")