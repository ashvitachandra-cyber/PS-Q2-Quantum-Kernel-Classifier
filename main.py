"""
PS-Q2: High-Dimensional Quantum Kernel Classifier
ZZFeatureMap + FidelityQuantumKernel + QSVC on benchmark data.
"""

import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel

RANDOM_STATE = 42
N_SAMPLES = 100  # small stratified subset keeps simulator runtime manageable
N_FEATURES = 4
OUTPUT_DIR = "outputs"


def score_model(name, y_true, y_pred):
    """Return common binary classification metrics."""
    result = {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # 1. Load benchmark data and choose a reproducible stratified subset.
    dataset = load_breast_cancer()
    X_all = dataset.data
    y_all = dataset.target
    rng = np.random.default_rng(RANDOM_STATE)

    chosen = []
    for label in np.unique(y_all):
        class_indices = np.flatnonzero(y_all == label)
        n_class = N_SAMPLES // 2
        chosen.extend(rng.choice(class_indices, size=n_class, replace=False).tolist())
    chosen = np.array(chosen)
    rng.shuffle(chosen)
    X = X_all[chosen]
    y = y_all[chosen]

    # 2. Split before fitting transformations to avoid data leakage.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    # 3. Scale using training data, then reduce to four components.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    pca = PCA(n_components=N_FEATURES, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Quantum rotation angles are scaled to [0, pi].
    angle_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train = angle_scaler.fit_transform(X_train_pca)
    X_test = angle_scaler.transform(X_test_pca)
    X_test = np.clip(X_test, 0.0, np.pi)

    # 4. Define ZZFeatureMap and fidelity quantum kernel.
    feature_map = ZZFeatureMap(
        feature_dimension=N_FEATURES,
        reps=2,
        entanglement="linear",
    )
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

    # 5. Train QSVC and predict test labels.
    print("Training QSVC. Quantum kernel evaluation may take a few minutes...")
    qsvc = QSVC(quantum_kernel=quantum_kernel)
    qsvc.fit(X_train, y_train)
    qsvc_predictions = qsvc.predict(X_test)

    qsvc_metrics = score_model("Quantum QSVC", y_test, qsvc_predictions)
    print("\n=== Quantum QSVC results ===")
    print(classification_report(
        y_test, qsvc_predictions, target_names=dataset.target_names, zero_division=0
    ))
    print(f"Accuracy:  {qsvc_metrics['accuracy']:.4f}")
    print(f"Precision: {qsvc_metrics['precision']:.4f}")
    print(f"Recall:    {qsvc_metrics['recall']:.4f}")
    print(f"F1-score:  {qsvc_metrics['f1_score']:.4f}")

    # 6. Classical baseline for context.
    classical_svm = SVC(kernel="rbf", random_state=RANDOM_STATE)
    classical_svm.fit(X_train, y_train)
    classical_predictions = classical_svm.predict(X_test)
    classical_metrics = score_model("Classical RBF SVM", y_test, classical_predictions)

    print("\n=== Classical RBF SVM baseline ===")
    print(f"Accuracy:  {classical_metrics['accuracy']:.4f}")
    print(f"Precision: {classical_metrics['precision']:.4f}")
    print(f"Recall:    {classical_metrics['recall']:.4f}")
    print(f"F1-score:  {classical_metrics['f1_score']:.4f}")

    # 7. Save QSVC confusion matrix.
    cm = confusion_matrix(y_test, qsvc_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.target_names)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("QSVC Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_qsvc.png"), dpi=160)
    plt.close()

    # 8. Save quantum kernel matrix for a small sample to visualize similarity.
    # Kernel values are calculated by the same fidelity kernel used by QSVC.
    n_visual = min(25, len(X_train))
    kernel_matrix = quantum_kernel.evaluate(x_vec=X_train[:n_visual])
    plt.figure(figsize=(7, 6))
    plt.imshow(kernel_matrix, interpolation="nearest", aspect="auto", cmap="viridis")
    plt.colorbar(label="Quantum fidelity")
    plt.title("Quantum Kernel Matrix (training subset)")
    plt.xlabel("Sample index")
    plt.ylabel("Sample index")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "quantum_kernel_matrix.png"), dpi=160)
    plt.close()

    # 9. Compare metrics in a bar chart.
    metric_names = ["accuracy", "precision", "recall", "f1_score"]
    x_positions = np.arange(len(metric_names))
    width = 0.36
    plt.figure(figsize=(8, 5))
    plt.bar(x_positions - width / 2, [qsvc_metrics[m] for m in metric_names],
            width, label="Quantum QSVC")
    plt.bar(x_positions + width / 2, [classical_metrics[m] for m in metric_names],
            width, label="Classical RBF SVM")
    plt.xticks(x_positions, ["Accuracy", "Precision", "Recall", "F1-score"])
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Quantum QSVC vs Classical SVM")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=160)
    plt.close()

    # 10. Persist real results from this run.
    output = {
        "dataset": "Wisconsin Breast Cancer (scikit-learn)",
        "samples_used": int(len(y)),
        "features_after_pca": N_FEATURES,
        "feature_map": "ZZFeatureMap(reps=2, entanglement='linear')",
        "kernel": "FidelityQuantumKernel",
        "random_state": RANDOM_STATE,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "qsvc": qsvc_metrics,
        "classical_rbf_svm": classical_metrics,
        "qsvc_confusion_matrix": cm.tolist(),
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved plots and metrics in: {OUTPUT_DIR}/")
    print("Files: confusion_matrix_qsvc.png, quantum_kernel_matrix.png, "
          "model_comparison.png, metrics.json")


if __name__ == "__main__":
    main()
