from sklearn.metrics import (precision_score,recall_score,f1_score,accuracy_score)
from src.visualization.plots import (plot_roc_curve,plot_precision_recall_curve,
                                     plot_confusion_matrix)

def evaluate_model(model, x, y, prefix):
    """
    Evaluate a classification model and generate metrics & plots.

    Args:
        model : Trained model with predict & predict_proba.
        x : Feature matrix for evaluation.
        y : True labels corresponding to x.
        prefix : Prefix for metric & figure keys (e.g., "train" or "val").

    Returns:
        Tuple of (metrics dict, figures dict).
    """
    y_pred = model.predict(x)
    y_prob = model.predict_proba(x)[:, 1]

    roc_auc, roc_fig = plot_roc_curve(y, y_prob)
    ap, pr_fig = plot_precision_recall_curve(y, y_prob)
    cm_fig = plot_confusion_matrix(
        y,
        y_pred,
        title=f"{prefix.capitalize()} Confusion Matrix")

    metrics = {
        f"{prefix}_roc_auc": roc_auc,
        f"{prefix}_average_precision": ap,
        f"{prefix}_precision": precision_score(y, y_pred),
        f"{prefix}_recall": recall_score(y, y_pred),
        f"{prefix}_f1_score": f1_score(y, y_pred),
        f"{prefix}_accuracy": accuracy_score(y, y_pred)}

    figures = {
        f"{prefix}_roc_curve.png": roc_fig,
        f"{prefix}_precision_recall_curve.png": pr_fig,
        f"{prefix}_confusion_matrix.png": cm_fig}

    return metrics, figures
