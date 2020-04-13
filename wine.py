#%%
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.exceptions import DataConversionWarning

import seaborn as sns

#warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings('ignore')

#%%
path_data = Path('data')
path_dataraw = path_data / 'raw'

#%%
df_raw = pd.read_csv(path_dataraw / 'winequalityN.csv')

# Filter DataFrame to just whitewine
df_wine = df_raw[df_raw.type == 'white']

# Clean Column Names
df_wine.columns = df_wine.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')


#%%
df_wine['quality_label'] = df_wine.quality.apply(lambda q: 0 if q < 7 else 1)

#%%
df_wine.head()
#%%
df_wine.isnull().sum()

#%%
df_wine.quality_label.value_counts()

#%%
df_wine.dropna().quality_label.value_counts()

#%%
df_wine.dropna(inplace=True)

#%%
# Split Data into training and testing datasets
X = df_wine[['alcohol', 'citric_acid', 'residual_sugar', 'free_sulfur_dioxide', 'density', 'ph', 'chlorides', 'sulphates', 'fixed_acidity']]
#X = df_wine.drop(['quality', 'quality_label', 'type'], axis=1)
y = df_wine.quality_label

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=40)

#%%
# models = [('scaler', preprocessing.StandardScaler()),
#                      ('RFR', RandomForestRegressor()),
#                      ('LR', LogisticRegression()),
#                      ('SVM', SVC())]

models = [('scaler', StandardScaler()),
          ('LR', LogisticRegression()),
          ('RFR', RandomForestClassifier(n_estimators=100))]

pipeline = Pipeline(models)

#%%
# hyperparameters = {'RFR__max_features': ['auto', 'sqrt', 'log2'],
#                    'RFR__max_depth': [None, 5, 3, 1],
#                    'RFR__n_estimators': [10, 100, 1000],
#                    'LR__penalty': ['l2', 'l1'],
#                    'LR__C': np.logspace(0,4,10),
#                    'SVM__C':[0.001,0.1,10,100,10e5],
#                    'SVM__gamma':[0.1,0.01]}


#hyperparameters = {'SVM__C':[0.001,0.1,10,100,10e5],
#                   'SVM__gamma':[0.1,0.01]}

hyperparameters = {'LR__penalty': ['l2', 'l1'],
                   'LR__C': np.logspace(0,4,10),
                   'RFR__max_features': ['auto', 'sqrt', 'log2'],
                   'RFR__max_depth': [None, 5, 3, 1]}

#%%
clf = GridSearchCV(pipeline, hyperparameters, cv=10)

clf.fit(X_train, y_train)

#%%
#pred = clf.predict(X_test)
result = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
print(result.mean())

#%%
print("score = %3.2f" %(grid.score(X_test,y_test)))
print(grid.best_params_)

#%%
best_model = gridsearch.fit(features, target)

#%%
# Construct some pipelines
pipe_lr = Pipeline([('scl', StandardScaler()),
			('clf', LogisticRegression(random_state=42))])

pipe_lr_pca = Pipeline([('scl', StandardScaler()),
			('pca', PCA(n_components=4)),
			('clf', LogisticRegression(random_state=42))])

pipe_rf = Pipeline([('scl', StandardScaler()),
			('clf', RandomForestClassifier(random_state=42))])

pipe_rf_pca = Pipeline([('scl', StandardScaler()),
			('pca', PCA(n_components=4)),
			('clf', RandomForestClassifier(random_state=42))])

pipe_svm = Pipeline([('scl', StandardScaler()),
			('clf', SVC(random_state=42))])

pipe_svm_pca = Pipeline([('scl', StandardScaler()),
			('pca', PCA(n_components=4)),
			('clf', SVC(random_state=42))])

#%%
# Set grid search params
param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
param_range_fl = [1.0, 0.5, 0.1]

grid_params_lr = [{'clf__penalty': ['l1', 'l2'],
		'clf__C': [1.0, 0.5, 0.1],
		'clf__solver': ['liblinear']}]

grid_params_rf = [{'clf__criterion': ['gini', 'entropy'],
		'clf__min_samples_leaf': param_range,
		'clf__max_depth': param_range,
		'clf__min_samples_split': param_range[1:]}]

grid_params_svm = [{'clf__kernel': ['linear', 'rbf'],
		'clf__C': param_range}]

#%%
# Construct grid searches
jobs = -1

scoring_tests = 'precision'

gs_lr = GridSearchCV(estimator=pipe_lr,
			param_grid=grid_params_lr,
			scoring=scoring_tests,
			cv=10) 

gs_lr_pca = GridSearchCV(estimator=pipe_lr_pca,
			param_grid=grid_params_lr,
			scoring=scoring_tests,
			cv=10)

gs_rf = GridSearchCV(estimator=pipe_rf,
			param_grid=grid_params_rf,
			scoring=scoring_tests,
			cv=10, 
			n_jobs=jobs)

gs_rf_pca = GridSearchCV(estimator=pipe_rf_pca,
			param_grid=grid_params_rf,
			scoring=scoring_tests,
			cv=10, 
			n_jobs=jobs)

gs_svm = GridSearchCV(estimator=pipe_svm,
			param_grid=grid_params_svm,
			scoring=scoring_tests,
			cv=10,
			n_jobs=jobs)

gs_svm_pca = GridSearchCV(estimator=pipe_svm_pca,
			param_grid=grid_params_svm,
			scoring=scoring_tests,
			cv=10,
			n_jobs=jobs)

#%%
# List of pipelines for ease of iteration
grids = [gs_lr, gs_lr_pca, gs_rf, gs_rf_pca, gs_svm, gs_svm_pca]

#%%
# Dictionary of pipelines and classifier types for ease of reference
grid_dict = {0: 'Logistic Regression', 1: 'Logistic Regression w/PCA', 
		2: 'Random Forest', 3: 'Random Forest w/PCA', 
		4: 'Support Vector Machine', 5: 'Support Vector Machine w/PCA'}

#%%
print('Performing model optimizations...')
best_acc = 0.0
best_clf = 0
best_gs = ''
for idx, gs in enumerate(grids):
    print('\nEstimator: %s' % grid_dict[idx])
    # Fit grid search
    gs.fit(X_train, y_train)
    # Best params
    print('Best params: %s' % gs.best_params_)
    # Best training data accuracy
    print('Best training accuracy: %.3f' % gs.best_score_)
    # Predict on test data with best params
    y_pred = gs.predict(X_test)
    # Test data accuracy of model with best params
    print('Test set accuracy score for best params: %.3f ' % accuracy_score(y_test, y_pred))
    print('Test set precision score for best params: %.3f ' % precision_score(y_test, y_pred))
    print('Test set recall score for best params: %.3f ' % recall_score(y_test, y_pred))
    # Track best (highest test accuracy) model
    if accuracy_score(y_test, y_pred) > best_acc:
        best_acc = accuracy_score(y_test, y_pred)
        best_gs = gs
        best_clf = idx
print('\nClassifier with best test set accuracy: %s' % grid_dict[best_clf])

#%%
gs_rf.fit(X_train, y_train)
results = gs.cv_results_

#%%
results

#%%
