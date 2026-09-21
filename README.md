FIFA Player Scouting ML Project


What's in this repo
machine_learning_final_project.ipynb – the full notebook
Fifa.csv – the dataset (you'll need to add this yourself, see below)
results.json – exported best hyperparameters and cross-validation stability scores
What we did

1. EDA — checked missing values, looked at the distribution of Value Per M$ (it was very skewed, so we log-transformed it), checked which numerical features correlate most with value, and looked at average ratings per position.

2. Preprocessing — split 80/20 before touching anything (to avoid leakage), one-hot encoded categorical columns, scaled numerical features with StandardScaler, and handled outliers in the target with IQR/winsorization.

3. Created classification targets — split players into 4 performance tiers (Low / Mid / High / Elite) based on Overall Rating, using percentiles to justify the thresholds.

4. Polynomial Regression — tried predicting value with polynomial features of different degrees.

5. Logistic Regression — baseline classifier for performance tier (without using Overall Rating, since that would be leakage).

6. Naive Bayes — compared GaussianNB, BernoulliNB, and ComplementNB, and also checked whether scaling affects GaussianNB (it doesn't, since it's just a linear rescaling).

7. Cross-validation — K-Fold for regression, Stratified K-Fold for classification, to check model stability and overfitting.

8. Unified Scouting System — the main part. Each of us took a model:

KNN (regression + classification)
SVM (SVR + SVC)
Random Forest (regression + classification)
Voting Ensemble combining all three

All models were tuned with GridSearchCV, and we compared everything against our Assignment 2 baselines.

Results
Model	Regression (R²)	Classification (Accuracy)
Baseline (Linear/Logistic)	0.5154	0.8728
KNN	0.8606	0.8892
SVM	0.8984	0.9118
Random Forest	0.9364	0.9014
Voting Ensemble	0.9159	0.9082

The Voting Ensemble was the most stable across cross-validation folds (R² = 0.9155 ± 0.0207, Accuracy = 0.9059 ± 0.0047), even though Random Forest alone scored slightly higher on regression — the ensemble trades a bit of peak performance for more consistent results.

Best hyperparameters found

See results.json for the full list, found via GridSearchCV:

KNN Regressor: n_neighbors=5, weights=distance, metric=euclidean
KNN Classifier: n_neighbors=9, weights=uniform, metric=euclidean
SVR / SVC: C=10.0, kernel=rbf, gamma=scale
RF Regressor: n_estimators=100, max_depth=10, min_samples_split=2
RF Classifier: n_estimators=200, max_depth=None, min_samples_split=10
How to run
bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn pycountry_convert

Put Fifa.csv in the same folder as the notebook, then run all cells top to bottom.
