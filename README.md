# PS-Q2: High-Dimensional Quantum Kernel Classifier

**Hackathon problem statement:** ZZFeatureMap Support Vector Machine (QSVC) on Benchmark Data

## Overview
This project implements a quantum-kernel classifier using Qiskit Machine Learning. It encodes classical benchmark data with a `ZZFeatureMap`, computes a fidelity-based quantum kernel, and trains a Quantum Support Vector Classifier (QSVC).

The Wisconsin Breast Cancer dataset from scikit-learn is used as the benchmark. To keep statevector simulation manageable, the workflow uses a stratified subset and reduces the input to four features using standardization and PCA.

## Pipeline
1. Load the benchmark dataset.
2. Create a stratified subset for a faster demo.
3. Standardize features and reduce them to four dimensions with PCA.
4. Scale features to the range `[0, π]`.
5. Split data into training and test sets.
6. Build a four-qubit `ZZFeatureMap`.
7. Evaluate the fidelity quantum kernel and train QSVC.
8. Evaluate accuracy, precision, recall, F1-score and confusion matrix.
9. Compare against a classical RBF SVM.
10. Save plots and a JSON metrics file to `outputs/`.

## Project files
- `main.py` — end-to-end implementation.
- `requirements.txt` — Python dependencies.
- `outputs/` — created automatically when the program runs.
- `PS-Q2_QSVC.ipynb` — notebook version of the workflow.

## Run locally
Use Python 3.10 or 3.11 in a virtual environment.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

The first run may take a few minutes because quantum kernel entries require circuit simulation.

## Run in Google Colab
1. Open Google Colab and create a new notebook.
2. Upload `main.py` or paste its contents into a cell.
3. Run `!pip install -r requirements.txt` after uploading `requirements.txt`.
4. Run `!python main.py`.

## Expected outputs
The program prints test metrics for QSVC and a classical RBF SVM, then saves:
- `outputs/confusion_matrix_qsvc.png`
- `outputs/quantum_kernel_matrix.png`
- `outputs/model_comparison.png`
- `outputs/metrics.json`

**Important:** Actual metric values are generated when you run the code; they are not hard-coded. Quantum simulation performance depends on package versions and runtime.

## Method
For input samples \(x_i\) and \(x_j\), the fidelity quantum kernel is conceptually:

`K(x_i, x_j) = |<φ(x_i)|φ(x_j)>|²`

The feature map `φ(x)` is implemented with a ZZ interaction feature map. QSVC uses this kernel in an SVM classifier.

## Limitations
- This is a quantum-circuit simulation on a classical computer, not execution on a real quantum processor.
- Four input features are used to limit circuit and kernel computation costs.
- Results do not by themselves demonstrate a quantum advantage over classical methods.
- The random subset and train/test split use fixed seeds for reproducibility.

## References
- Qiskit Machine Learning quantum kernel tutorial: https://qiskit-community.github.io/qiskit-machine-learning/tutorials/03_quantum_kernel.html
- Qiskit Machine Learning QSVC implementation: https://github.com/qiskit-community/qiskit-machine-learning/blob/main/qiskit_machine_learning/algorithms/classifiers/qsvc.py
