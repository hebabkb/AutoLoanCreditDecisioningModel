
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, auc, roc_curve, roc_auc_score
)
from sklearn.model_selection import (
    GridSearchCV, cross_val_score, cross_val_predict, StratifiedKFold
)
from sklearn.inspection import permutation_importance

def cross_val_pr_curve(model, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    mean_recall = np.linspace(0, 1, 100)
    mean_precision_sum = np.zeros_like(mean_recall)
    auc_scores = []

    X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
    y = y.to_numpy() if isinstance(y, pd.Series) else y

    for train_idx, val_idx in skf.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        y_scores = model.predict_proba(X[val_idx])[:, 1]
        precision, recall, _ = precision_recall_curve(y[val_idx], y_scores)
        auc_scores.append(auc(recall, precision))
        mean_precision_sum += np.interp(mean_recall, np.flip(recall), np.flip(precision))

    mean_precision = mean_precision_sum / cv
    plt.plot(mean_recall, mean_precision, label=f"Mean PR AUC={np.mean(auc_scores):.2f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve")
    plt.legend(); plt.grid(); plt.show()
    print(f"Mean PR AUC: {np.mean(auc_scores):.4f}")

def cross_val_roc_curve(model, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr_sum = np.zeros_like(mean_fpr)
    auc_scores = []

    X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
    y = y.to_numpy() if isinstance(y, pd.Series) else y

    for train_idx, val_idx in skf.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        y_scores = model.predict_proba(X[val_idx])[:, 1]
        fpr, tpr, _ = roc_curve(y[val_idx], y_scores)
        auc_scores.append(roc_auc_score(y[val_idx], y_scores))
        mean_tpr_sum += np.interp(mean_fpr, fpr, tpr)

    mean_tpr = mean_tpr_sum / cv
    plt.plot(mean_fpr, mean_tpr, label=f"Mean ROC AUC={np.mean(auc_scores):.2f}", color='darkorange')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend(); plt.grid(); plt.show()
    print(f"Mean ROC AUC: {np.mean(auc_scores):.4f}")

def plot_confusion_matrix(y_true, y_pred):
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap='Blues', normalize='true')
    plt.title("Normalized Confusion Matrix")
    plt.grid(False)
    plt.show()

def tune_hyperparameters(X_train, y_train, param_grid, model, metric):
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring=metric)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    return best_params

def evaluate_models(model, X, y, metric='precision'):
    scores = cross_val_score(model, X, y, cv=3, scoring=metric)
    print("CV Scores:", scores)
    print("Mean CV Score:", np.mean(scores))

    y_pred = cross_val_predict(model, X, y, cv=3)
    print("Classification Report:", classification_report(y, y_pred))

    plot_confusion_matrix(y, y_pred)
    cross_val_pr_curve(model, X, y)
    cross_val_roc_curve(model, X, y)

    return pd.DataFrame(classification_report(y, y_pred, output_dict=True)).T

def feature_importance(model, X, y, metric='accuracy'):
    result = permutation_importance(model, X, y, scoring=metric, n_repeats=10, random_state=42)
    importances = pd.DataFrame({'Feature': X.columns, 'Importance': result.importances_mean})
    importances = importances.sort_values(by='Importance', ascending=False)
    importances.plot(x='Feature', y='Importance', kind='bar', figsize=(10, 5), legend=False, title='Feature Importance')
    plt.show()
    return importances
    
def threshold_analysis(y_true, y_probs, thresholds=np.arange(0.1, 0.9, 0.05)):
    rows = []
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        rows.append({
            'Threshold': t,
            'Precision': report['1']['precision'],
            'Recall': report['1']['recall'],
            'F1': report['1']['f1-score']
        })
    
    df = pd.DataFrame(rows)
    df.plot(x='Threshold', y=['Precision', 'Recall', 'F1'], marker='o', title='Threshold vs Metrics')
    plt.grid(); plt.show()
    return df
