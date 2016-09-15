### http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter ###
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import KFold
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV

clfs = [
  DecisionTreeClassifier(),
  GaussianNB(),
  MultinomialNB(),
  BernoulliNB(),
  KNeighborsClassifier(),
  RadiusNeighborsClassifier(),
  SVC(),
  NuSVC(),
  RandomForestClassifier()
]

exit()

ham_txt = json.load(open('dataset_dev/ham_dev.json'))
spam_txt = json.load(open('dataset_dev/spam_dev.json'))
validation_indices = open('indices_validation.in').read().split()

ham_train = [ham_txt[i] for i in range(0, len(ham_txt)) if validation_indices[i] == '0']
spam_train = [spam_txt[i] for i in range(0, len(spam_txt)) if validation_indices[i] == '0']
ham_validation = [ham_txt[i] for i in range(0, len(ham_txt)) if validation_indices[i] == '1']
spam_validation = [spam_txt[i] for i in range(0, len(spam_txt)) if validation_indices[i] == '1']

X_train = ham_train + spam_train
y_train = ['ham' for _ in range(len(ham_train))] + ['spam' for _ in range(len(spam_train))]
X_validation = ham_validation + spam_validation
y_validation = ['ham' for _ in range(len(ham_validation))] + ['spam' for _ in range(len(spam_validation))]

kf = KFold(len(X_train), n_folds=10)
for train_index, test_index in kf:
  # Index to create the folds
  kf_X_train = []
  kf_X_test = []
  kf_y_train = []
  kf_y_test = []
  for i in train_index:
    kf_X_train.append(X_train[i])
    kf_y_train.append(y_train[i])
  for i in test_index:
    kf_X_test.append(X_train[i])
    kf_y_test.append(y_train[i])

  vectorizer = CountVectorizer(token_pattern='[^\d\W_][\w|\']+', max_features=500)
  X = vectorizer.fit_transform(kf_X_train)
  X_test = vectorizer.transform(kf_X_test)

  for i in len(param_grids):
    grid_search = GridSearchCV(clfs[i], param_grid=param_grids[i])
    grid_search.fit(X, kf_y_train, scoring='f1', cv=ShuffleSplit(1, test_size=0.20, n_iter=1, random_state=0))
    print i
    print grid_search.best_estimator_
    print grid_search.best_score_
    print grid_search.best_params_
    print grid_search.scorer_

  # clf = DecisionTreeClassifier()
  # clf.fit(X, kf_y_train)
  # print clf.score(X_test, kf_y_test)
  # y_predicted = clf.predict(X_test)
  # print accuracy_score(kf_y_test, y_predicted)
