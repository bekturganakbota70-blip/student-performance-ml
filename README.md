# Student Performance Prediction (ML Algorithms - Midterm Project
## Group SE-2431

## Team and roles
| Member | Role / technical contribution |
|---|---|
| Akbota | Data acquisition and loading, data cleaning (outliers, dropout analysis), exploratory data analysis (5 plots and interpretations) |
| Nurzhaina | Feature engineering (`parents_edu`, `alc_total`, scaling, encoding), stratified train/validation/test split, leakage prevention, baseline model |
| Diana | Decision Tree, KNN and SVM models, cross-validation, metrics, error analysis (confusion matrices, feature importance), open problems and final-stage plan |

## Project question
Can we predict whether a secondary-school student passes the final mathematics exam (G3 >= 10 on a 0-20 scale) and identify students at risk of failing?

- **Task:** binary classification. **Target:** `pass` (1 = pass, 0 = fail).
- **Success metric:** F1 and recall for the "fail" class, plus ROC-AUC. Baseline: always predict "pass" (accuracy 0.671).
- **Two scenarios:** A uses previous grades G1 and G2; B excludes them (realistic for early warning).

## Dataset
- **Source:** [UCI Machine Learning Repository - Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) (Cortez & Silva, 2008). File used: `student-mat.csv`.
- **Size:** 395 students, 33 columns. **License:** CC BY 4.0.
- **Known issues:** small size, class imbalance (67% / 33%), 38 students with G3 = 0 (likely dropouts, all with absences = 0), noisy `absences`.

## How to run
1. Open the notebook in Google Colab: [Open in Colab](https://colab.research.google.com/github/bekturganakbota70-blip/student-performance-ml/blob/main/student_performance_midterm.ipynb)
2. Choose **Runtime -> Run all**. The notebook downloads the dataset from UCI automatically (internet access required).
3. Local run: `pip install pandas scikit-learn matplotlib seaborn`, then open the notebook in Jupyter.

## Current results (validation set, 79 students, 26 of them "fail")
| Scenario | Model | Accuracy | F1 (fail) | ROC-AUC | CV F1 (fail) |
|---|---|---|---|---|---|
| A: with G1, G2 | Baseline | 0.671 | 0.000 | 0.500 | 0.000 |
| A | Decision Tree | 0.873 | 0.783 | 0.833 | 0.832 |
| A | KNN (k=5) | 0.797 | 0.600 | 0.841 | 0.587 |
| A | SVM (RBF) | 0.899 | 0.826 | 0.964 | 0.776 |
| B: without G1, G2 | Decision Tree | 0.671 | 0.409 | 0.671 | 0.435 |
| B | KNN (k=5) | 0.722 | 0.353 | 0.563 | 0.367 |
| B | SVM (RBF) | 0.747 | 0.375 | 0.646 | 0.341 |

**Key findings**
- Previous grades G1 and G2 carry most of the predictive power; without them all models are close to the baseline.
- Errors in Scenario A concentrate around the pass threshold (G3 = 8-10).
- Zero absences mix full attendance and unrecorded dropouts, so `absences` is noisy.
- The validation set is small, so all comparisons are preliminary. The test set has not been used.

## Next steps (final stage)
Hyperparameter tuning (GridSearchCV), class weights and threshold tuning to improve recall for "fail", feature selection, models with and without the G3 = 0 students, a regression / three-class risk formulation, repeated cross-validation, fitting the `absences` cap inside the pipeline, and careful use of the Portuguese-course file (overlapping students) with a student-level split.
