# ============================================================
# PREDICTIVE MAINTENANCE - MACHINE FAILURE PREDICTION
# ML Modelling Script
# Dataset: AI4I 2020 Predictive Maintenance Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("=" * 60)
print("STEP 1: LOADING DATA")
print("=" * 60)

df = pd.read_csv('predictive_maintenance.csv')

print(f"Dataset Shape   : {df.shape}")
print(f"Total Records   : {len(df)}")
print(f"Total Features  : {df.shape[1]}")
print(f"\nColumn Names    : {df.columns.tolist()}")
print(f"\nMissing Values  :\n{df.isnull().sum()}")
print(f"\nTarget Distribution:\n{df['Target'].value_counts()}")
print(f"\nFailure Rate    : {df['Target'].mean()*100:.2f}%")


# ============================================================
# STEP 2: DATA PREPARATION & FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: DATA PREPARATION & FEATURE ENGINEERING")
print("=" * 60)

# Convert temperature from Kelvin to Celsius
df['Air_Temp_C']     = df['Air temperature [K]'] - 273.15
df['Process_Temp_C'] = df['Process temperature [K]'] - 273.15

# Engineer new features
df['Temp_Diff'] = df['Process_Temp_C'] - df['Air_Temp_C']
df['Power']     = df['Torque [Nm]'] * df['Rotational speed [rpm]'] * (2 * np.pi / 60)

# Encode machine type (L=1, M=2, H=0)
le = LabelEncoder()
df['Type_enc'] = le.fit_transform(df['Type'])

print("Feature Engineering Complete:")
print("  ✓ Air Temp converted to Celsius")
print("  ✓ Process Temp converted to Celsius")
print("  ✓ Temp Diff created (Process - Air)")
print("  ✓ Power created (Torque × RPM)")
print("  ✓ Machine Type encoded (L/M/H → numeric)")

# Define features and target
features = [
    'Type_enc',
    'Air_Temp_C',
    'Process_Temp_C',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
    'Temp_Diff',
    'Power'
]

X = df[features]
y = df['Target']

print(f"\nFeatures used   : {features}")
print(f"Target variable : Target (0=Healthy, 1=Failed)")


# ============================================================
# STEP 3: TRAIN/TEST SPLIT
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: TRAIN / TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y        # preserve failure ratio in both sets
)

print(f"Training Set    : {X_train.shape[0]} records (80%)")
print(f"Test Set        : {X_test.shape[0]} records (20%)")
print(f"Failures in Train: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"Failures in Test : {y_test.sum()} ({y_test.mean()*100:.2f}%)")


# ============================================================
# STEP 4: TRAIN RANDOM FOREST MODEL
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: TRAINING RANDOM FOREST MODEL")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=100,       # 100 decision trees
    random_state=42,        # reproducibility
    class_weight='balanced' # handle imbalanced classes
)

rf_model.fit(X_train, y_train)
print("Model Training  : COMPLETE")
print(f"Number of Trees : {rf_model.n_estimators}")
print(f"Class Weight    : balanced (handles 3.39% failure rate)")


# ============================================================
# STEP 5: MODEL EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: MODEL EVALUATION")
print("=" * 60)

y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc      = roc_auc_score(y_test, y_prob)

print(f"Accuracy        : {accuracy*100:.2f}%")
print(f"ROC-AUC Score   : {auc:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Healthy', 'Failed']))


# ============================================================
# STEP 6: FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: FEATURE IMPORTANCE")
print("=" * 60)

feat_labels = [
    'Machine Type', 'Air Temp', 'Process Temp',
    'Rotational Speed', 'Torque', 'Tool Wear',
    'Temp Diff', 'Power'
]

importances = rf_model.feature_importances_
feat_df = pd.DataFrame({
    'Feature'   : feat_labels,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print(feat_df.to_string(index=False))


# ============================================================
# STEP 7: DISAGGREGATED EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: DISAGGREGATED EVALUATION")
print("=" * 60)

from sklearn.metrics import precision_score, recall_score, f1_score

# Add predictions to test set
X_test_copy = X_test.copy()
X_test_copy['y_true'] = y_test.values
X_test_copy['y_pred'] = y_pred

# Decode machine type back
X_test_copy['Type'] = le.inverse_transform(X_test_copy['Type_enc'].astype(int))

# --- Unitary Results: By Machine Type ---
print("\nUnitary Results — By Machine Type:")
print("-" * 50)
for t in ['L', 'M', 'H']:
    mask = X_test_copy['Type'] == t
    if mask.sum() == 0:
        continue
    yt = X_test_copy.loc[mask, 'y_true']
    yp = X_test_copy.loc[mask, 'y_pred']
    p  = precision_score(yt, yp, zero_division=0)
    r  = recall_score(yt, yp, zero_division=0)
    f  = f1_score(yt, yp, zero_division=0)
    a  = accuracy_score(yt, yp)
    print(f"  {t} Quality | Accuracy: {a:.2%} | Precision: {p:.2%} | Recall: {r:.2%} | F1: {f:.2%}")

# --- Unitary Results: By Tool Wear ---
print("\nUnitary Results — By Tool Wear Category:")
print("-" * 50)
bins  = [0, 50, 100, 150, 200, 300]
lbls  = ['0-50', '51-100', '101-150', '151-200', '200+']
X_test_copy['TW_bin'] = pd.cut(
    X_test_copy['Tool wear [min]'], bins=bins, labels=lbls)

for b in lbls:
    mask = X_test_copy['TW_bin'] == b
    if mask.sum() == 0:
        continue
    yt = X_test_copy.loc[mask, 'y_true']
    yp = X_test_copy.loc[mask, 'y_pred']
    p  = precision_score(yt, yp, zero_division=0)
    r  = recall_score(yt, yp, zero_division=0)
    f  = f1_score(yt, yp, zero_division=0)
    a  = accuracy_score(yt, yp)
    print(f"  {b} mins | Accuracy: {a:.2%} | Precision: {p:.2%} | Recall: {r:.2%} | F1: {f:.2%}")

# --- Intersectional Results ---
print("\nIntersectional Results — Machine Type + Tool Wear:")
print("-" * 50)
for t in ['L', 'H']:
    for b in ['0-50', '200+']:
        mask = (X_test_copy['Type'] == t) & (X_test_copy['TW_bin'] == b)
        if mask.sum() < 5:
            continue
        yt = X_test_copy.loc[mask, 'y_true']
        yp = X_test_copy.loc[mask, 'y_pred']
        r  = recall_score(yt, yp, zero_division=0)
        a  = accuracy_score(yt, yp)
        print(f"  {t} Quality + Tool Wear {b} mins | Accuracy: {a:.2%} | Recall: {r:.2%}")

# ============================================================
# STEP 8: GENERATE EVALUATION CHARTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: GENERATING EVALUATION CHARTS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('#0f1117')
fig.suptitle(
    'Random Forest — Model Evaluation\nAI4I 2020 Predictive Maintenance Dataset',
    color='white', fontsize=14, fontweight='bold', y=0.98
)

BG   = '#1a1d24'
BLUE = '#4C9BE8'
RED  = '#E74C3C'
GRN  = '#2ECC71'
GOLD = '#F39C12'
TXT  = 'white'

def style(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color=TXT, fontsize=12, fontweight='bold', pad=10)
    ax.tick_params(colors=TXT)
    for s in ax.spines.values(): s.set_edgecolor('#333')
    ax.xaxis.label.set_color(TXT)
    ax.yaxis.label.set_color(TXT)

# --- Chart 1: Confusion Matrix ---
ax1 = axes[0, 0]
cm  = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['Healthy','Failed'],
            yticklabels=['Healthy','Failed'],
            cbar=False, linewidths=1,
            annot_kws={'size':14,'color':'white','weight':'bold'})
style(ax1, 'Confusion Matrix')
ax1.set_xlabel('Predicted', fontsize=11)
ax1.set_ylabel('Actual', fontsize=11)

# --- Chart 2: ROC Curve ---
ax2 = axes[0, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax2.plot(fpr, tpr, color=BLUE, lw=2.5, label=f'AUC = {auc:.4f}')
ax2.plot([0,1],[0,1], color='#555', lw=1.5, linestyle='--', label='Random')
ax2.fill_between(fpr, tpr, alpha=0.15, color=BLUE)
style(ax2, 'ROC Curve')
ax2.set_xlabel('False Positive Rate', fontsize=11)
ax2.set_ylabel('True Positive Rate', fontsize=11)
leg = ax2.legend(fontsize=10, facecolor=BG, edgecolor='#333')
for t in leg.get_texts(): t.set_color(TXT)

# --- Chart 3: Feature Importance ---
ax3 = axes[1, 0]
fi_sorted = feat_df.sort_values('Importance')
colors    = [RED if v == feat_df['Importance'].max()
             else BLUE for v in fi_sorted['Importance']]
bars = ax3.barh(fi_sorted['Feature'], fi_sorted['Importance'],
                color=colors, edgecolor='none', height=0.6)
style(ax3, 'Feature Importance')
ax3.set_xlabel('Importance Score', fontsize=11)
for bar, val in zip(bars, fi_sorted['Importance']):
    ax3.text(val + 0.002, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', color=TXT, fontsize=9)

# --- Chart 4: Failure Rate by Tool Wear ---
ax4   = axes[1, 1]
bins  = [0, 50, 100, 150, 200, 300]
lbls  = ['0-50', '51-100', '101-150', '151-200', '200+']
df['TW_bin']  = pd.cut(df['Tool wear [min]'], bins=bins, labels=lbls)
fail_rate     = df.groupby('TW_bin', observed=True)['Target'].mean() * 100
bar_colors    = [GRN if v < 5 else GOLD if v < 10 else RED
                 for v in fail_rate.values]
bars2 = ax4.bar(fail_rate.index, fail_rate.values,
                color=bar_colors, edgecolor='none', width=0.6)
style(ax4, 'Failure Rate by Tool Wear (%)')
ax4.set_xlabel('Tool Wear (mins)', fontsize=11)
ax4.set_ylabel('Failure Rate (%)', fontsize=11)
for bar, val in zip(bars2, fail_rate.values):
    ax4.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.2,
             f'{val:.1f}%', ha='center',
             color=TXT, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
print("  ✓ model_evaluation.png saved")


# ============================================================
# STEP 8: SAVE MODEL
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: SAVING MODEL FILES")
print("=" * 60)

joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(le,       'label_encoder.pkl')
joblib.dump(features, 'features.pkl')

print("  ✓ rf_model.pkl       — trained model")
print("  ✓ label_encoder.pkl  — type encoder")
print("  ✓ features.pkl       — feature list")

print("\n" + "=" * 60)
print("MODELLING COMPLETE!")
print(f"  Final Accuracy : {accuracy*100:.2f}%")
print(f"  Final AUC      : {auc:.4f}")
print("=" * 60)
