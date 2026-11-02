# EEG-Based Epilepsy Detection with Machine Learning

**Tools:** Python, pandas, NumPy, scikit-learn, seaborn, matplotlib, Jupyter Notebook

**My role:** Data Analyst (sole author)

**Project type:** MSc coursework, MATH516 Machine Learning and Artificial Intelligence for Healthcare, University of Plymouth
**Data:** Public Kaggle dataset of EEG features from 198 subjects (99 with epilepsy, 99 without). Each subject has 40 pre-extracted features: power, fractal dimension and six wavelet coefficients for each of the five frequency bands (delta, theta, alpha, beta, gamma).

## Background
Epilepsy is a neurological condition that causes recurrent seizures. Diagnosis relies heavily on electroencephalography (EEG), which records the brain's electrical activity without surgery. Machine learning can pick up subtle EEG patterns that are hard to see by eye, which could support faster and more consistent diagnosis.

## Problem and aim
Build a classifier that separates people with epilepsy from healthy controls using EEG features, and answer five questions:
- Can machine learning and EEG detect epilepsy accurately (AUC above 0.8)?
- Which EEG features are most indicative of epilepsy?
- What is the smallest feature set that still performs well?
- How do different classifiers compare on performance, tuning effort, robustness, interpretability and training time?
- Which additional EEG features might improve detection?

## My approach
1. **Data preparation:** loaded the data with pandas, checked its structure and recoded the target (epilepsy = 1, no epilepsy = 0). Excluded the ID column. No missing values needed handling.
2. **Dimensionality reduction:** checked for no-variance and low-variance features, then removed redundant features using a Pearson correlation threshold of 0.80. Four features were dropped, leaving 36.
3. **Model selection:** compared 11 classifiers using 10-fold cross-validation on standardised features, and kept the top five: SVM, Random Forest, Extra Trees, Bagging and Logistic Regression (mean CV accuracy 0.990 to 1.000).
4. **Feature selection:** used Recursive Feature Elimination with Cross-Validation (RFECV), which kept 13 of the 36 features.
5. **Evaluation:** split the data 70:30 with balanced classes (60 test subjects), scaled after the split, and compared all five models on the 36 and 13 features using accuracy, sensitivity, specificity, precision, F1 and AUC. For the 13-feature models, I also looked at confusion matrices and ROC curves.

## Results

**Test set, all 36 features**

| Model | Accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---|---|
| SVM | 0.9833 | 1.0000 | 0.9667 | 0.9978 |
| Random Forest | 0.9833 | 0.9667 | 1.0000 | 1.0000 |
| Extra Trees | 0.9833 | 1.0000 | 0.9667 | 1.0000 |
| Bagging | 0.9833 | 0.9667 | 1.0000 | 1.0000 |
| Logistic Regression | 0.9833 | 1.0000 | 0.9667 | 0.9989 |

**Test set, 13 selected features**

| Model | Accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---|---|
| SVM | 0.9833 | 1.0000 | 0.9667 | 1.0000 |
| Random Forest | 0.9833 | 1.0000 | 0.9667 | 1.0000 |
| Extra Trees | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Bagging | 0.9833 | 0.9667 | 1.0000 | 1.0000 |
| Logistic Regression | 0.9667 | 0.9667 | 0.9667 | 0.9978 |


## Key findings
- **Accurate detection is possible.** All five models reached AUC above 0.99 and accuracy of 96.7% or higher on the test set, well above the 0.8 target.
- **A much smaller feature set held up.** Cutting from 36 to 13 features (about 64% fewer) kept all five models at 96.7% accuracy or higher. Extra Trees classified all 60 test subjects correctly, RF, SVM and Bagging made one error each, and Logistic Regression made two. With a test set this small, one subject is about 1.7 percentage points, so these differences are within chance.
- **Delta-band features dominated.** Of the 13 selected features, 8 are delta-band (power, fractal dimension and all six wavelet coefficients), 2 are alpha-band, 2 theta-band and 1 beta-band. None are gamma-band. The least useful features were mostly gamma and beta wavelet coefficients (cd5_g, cd5_b, cd6_g, cd4_g, cd4_b).
- **Classifier trade-offs.** The ensemble methods (RF, Extra Trees, Bagging) needed little tuning and were stable. The SVM needed tuning (a polynomial kernel scored about 0.85 until I switched to a linear kernel). Logistic Regression was the most interpretable and the fastest to train, with the SVM slowest.
- **Cross-validation agreed.** In 10-fold cross-validation the top models scored 0.990 to 1.000, with the SVM at 1.000.

## Limitations
- **Small sample.** 198 subjects and a 60-subject test set mean that results are imprecise (a perfect score on 60 subjects is still consistent with a true accuracy of roughly 94%), and the models can't be reliably ranked.
- **Feature selection used all the data.** RFECV was run on the full dataset before the train/test split, so the 13-feature results are likely optimistic. A nested or pipeline-based evaluation is the next step.
- **Single split.** Results come from one train/test split. Repeated cross-validation would give a steadier estimate.
- **Pre-extracted features.** I worked from extracted features, not raw EEG, so I couldn't check signal quality or artefacts.
- **Artificial class balance.** The data is 50% epilepsy, unlike real clinical populations, so real-world predictive values would differ.
- **Not a clinical tool.** This is a coursework model for healthy controls versus epilepsy, so it doesn't cover seizure timing, seizure types or other conditions.
- **The smallest useful feature set is not established.** 13 is the smallest set I tested.

## Recommendations and next steps
1. **Evaluate without leakage.** Move scaling and RFECV into a pipeline and use repeated stratified cross-validation, then report confidence intervals.
2. **Find the smallest useful feature set.** Rank features with permutation importance or Random Forest importance, then test the top 3, 5 and 8 features to see where performance drops.
3. **Validate externally.** Test on a separate, larger, multi-channel EEG dataset before any claim about generalisation.
4. **Add richer features.** Spectral entropy and phase synchronisation could capture signal complexity and connectivity that band power misses.
5. **Prioritise explainability for clinical use.** Clinicians are more likely to trust a model with clear explanations, so consider Logistic Regression or SHAP values on the tree models.
6. **Check for bias.** Review how performance varies by age, sex and epilepsy type before any real-world use.
7. **Treat it as decision support.** Any deployed version should assist clinicians, not replace them.

## What I learned
This project showed me how to compare classifiers, use feature selection to reduce dimensionality, and evaluate a healthcare model with several metrics. It also showed me how easily a small test set and a selection step done before splitting can make results look better than they are.


## Repository contents
- README.md: this write-up
- EEG-Based Epilepsy Detection Project.ipynb: the full analysis
- requirements.txt: Python packages
