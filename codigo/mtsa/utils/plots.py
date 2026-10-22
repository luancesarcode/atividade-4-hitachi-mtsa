import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def plot_roc_pr_curves(y, y_hat, figsize=(12, 5), save_path=None):
    roc_auc = roc_auc_score(y, y_hat)
    pr_auc = average_precision_score(y, y_hat)

    fpr, tpr, _ = roc_curve(y, y_hat)
    precision, recall, _ = precision_recall_curve(y, y_hat)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].plot(fpr, tpr, label=f"AUC-ROC = {roc_auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curve",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(recall, precision, label=f"AP = {pr_auc:.4f}")
    axes[1].set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision-Recall Curve",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return roc_auc, pr_auc
