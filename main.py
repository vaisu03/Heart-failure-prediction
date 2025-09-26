# ================================================================
# Heart Failure Prediction with AIW-PSO Optimized GBM + RandomForest Ensemble
# ================================================================

import pandas as pd
import numpy as np
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, cross_val_score
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ================================================================
# STEP 1: Load & Process Dataset
# ================================================================
df = pd.read_csv('data\heart_failure_clinical_records_dataset.csv')

# ✅ Log-transform skewed features
df['creatinine_phosphokinase'] = np.log1p(df['creatinine_phosphokinase'])
df['serum_creatinine'] = np.log1p(df['serum_creatinine'])

# ✅ Feature Engineering
df['creatinine_sodium_ratio'] = df['serum_creatinine'] / df['serum_sodium']
df['age_time_interaction'] = df['age'] * df['time']
df['ejection_time_ratio'] = df['ejection_fraction'] / (df['time'] + 1)

# ✅ Ensure binary columns are integers
binary_cols = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
df[binary_cols] = df[binary_cols].astype(int)

# ✅ Features & Target
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

# ✅ Remove low-variance features
vt = VarianceThreshold(threshold=0.01)
X = pd.DataFrame(vt.fit_transform(X), columns=X.columns[vt.get_support()])

# ✅ Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# ✅ Balance dataset using SMOTETomek
smote_tomek = SMOTETomek(random_state=42)
X_res, y_res = smote_tomek.fit_resample(X_scaled, y)
X_res = pd.DataFrame(X_res, columns=X.columns) 
print("✅ Dataset processed:", X_res.shape, "samples ready")

# ================================================================
# STEP 2: Train-Test Split
# ================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# ================================================================
# STEP 3: Custom AIW-PSO Implementation
# ================================================================
def fitness_gbm(params):
    """Fitness function for GBM optimization"""
    n_estimators = int(params[0])
    learning_rate = params[1]
    max_depth = int(params[2])
    subsample = params[3]
    
    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=42,
        n_jobs=-1
    )
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return 1 - scores.mean()  # minimize error

def aiw_pso(fitness_func, bounds, num_particles=30, max_iter=50, w_max=0.9, w_min=0.4, c1=2, c2=2):
    """Adaptive Inertia Weight Particle Swarm Optimization"""
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    X = np.random.uniform(lb, ub, (num_particles, dim))
    V = np.zeros((num_particles, dim))
    pbest = X.copy()
    pbest_score = np.array([fitness_func(x) for x in X])
    gbest_idx = np.argmin(pbest_score)
    gbest = pbest[gbest_idx].copy()
    gbest_score = pbest_score[gbest_idx]
    
    for t in range(max_iter):
        w = w_max - ((w_max - w_min) * t / max_iter)
        for i in range(num_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            V[i] = w * V[i] + c1 * r1 * (pbest[i] - X[i]) + c2 * r2 * (gbest - X[i])
            X[i] = np.clip(X[i] + V[i], lb, ub)
            score = fitness_func(X[i])
            if score < pbest_score[i]:
                pbest[i], pbest_score[i] = X[i].copy(), score
                if score < gbest_score:
                    gbest, gbest_score = X[i].copy(), score
        print(f"Iteration {t+1}/{max_iter}, Best CV Accuracy: {1-gbest_score:.4f}")
    return gbest, 1 - gbest_score

# ================================================================
# STEP 4: Run AIW-PSO Optimization
# ================================================================
bounds = [
    (50, 500),    # n_estimators
    (0.005, 0.5), # learning_rate
    (2, 15),      # max_depth
    (0.5, 1.0)    # subsample
]

best_pos, best_acc = aiw_pso(fitness_gbm, bounds)
print("✅ Best Hyperparameters (GBM):", best_pos, "CV Accuracy:", best_acc)

best_n, best_lr, best_depth, best_subsample = int(best_pos[0]), best_pos[1], int(best_pos[2]), best_pos[3]
gbm_model = LGBMClassifier(
    n_estimators=best_n, learning_rate=best_lr, max_depth=best_depth,
    subsample=best_subsample, random_state=42, n_jobs=-1
)
gbm_model.fit(X_train, y_train)

# ================================================================
# STEP 5: Ensemble Model (GBM + RandomForest)
# ================================================================
rf = RandomForestClassifier(n_estimators=400, max_depth=12, random_state=42, n_jobs=-1)

ensemble_model = VotingClassifier(
    estimators=[('gbm', gbm_model), ('rf', rf)],
    voting='soft'
)
ensemble_model.fit(X_train, y_train)

# ================================================================
# STEP 6: Model Evaluation
# ================================================================
y_pred = ensemble_model.predict(X_test)
print("\n📊 Ensemble Model Performance:")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("F1-Score:", round(f1_score(y_test, y_pred), 4))
print("ROC AUC:", round(roc_auc_score(y_test, y_pred), 4))

# ================================================================
# STEP 7: ROC Curve
# ================================================================
y_prob = ensemble_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'Ensemble (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0,1], [0,1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# ================================================================
# STEP 8: Feature Importance from GBM
# ================================================================
plt.figure(figsize=(8,5))
plt.title("Feature Importance (Optimized GBM)")
plt.show()

# ================================================================
# STEP 9: Save Model
# ================================================================
joblib.dump(ensemble_model, "heart_failure_model.pkl")
print("✅ Training Complete & Model saved as heart_failure_model.pkl")
