### http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter ###
import sys
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
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

X_train = []
y_train = []
X_validation = []
y_validation = []
X = []
X_val = []

def load_data():
  ham_txt = json.load(open('dataset_dev/ham_dev.json'))
  spam_txt = json.load(open('dataset_dev/spam_dev.json'))
  validation_indices = open('indices_validation.in').read().split()

  ham_train = [ham_txt[i] for i in range(0, len(ham_txt)) if validation_indices[i] == '0']
  spam_train = [spam_txt[i] for i in range(0, len(spam_txt)) if validation_indices[i] == '0']
  ham_validation = [ham_txt[i] for i in range(0, len(ham_txt)) if validation_indices[i] == '1']
  spam_validation = [spam_txt[i] for i in range(0, len(spam_txt)) if validation_indices[i] == '1']

  global X_train
  X_train = ham_train + spam_train
  global y_train
  y_train = [0 for _ in range(len(ham_train))] + [1 for _ in range(len(spam_train))]
  global X_validation
  X_validation = ham_validation + spam_validation
  global y_validation
  y_validation = [0 for _ in range(len(ham_validation))] + [1 for _ in range(len(spam_validation))]

  vectorizer = CountVectorizer(token_pattern='[^\d\W_][\w|\']+', max_features=500)
  global X
  X = vectorizer.fit_transform(X_train)
  global X_val
  X_val = vectorizer.transform(X_validation)

def etapa1():
  clf = DecisionTreeClassifier()
  param_grid = json.load(open('parameters/Trees/decision_tree_params.json'))
  grid_search = GridSearchCV(clf, param_grid=param_grid, scoring='f1', cv=10, n_jobs=-1)
  grid_search.fit(X, y_train)
  print 'Best Estimator:', str(grid_search.best_estimator_)
  print 'Best Score:', str(grid_search.best_score_)
  print 'Best Params:', str(grid_search.best_params_)
  print 'Validation Score:', grid_search.score(X_val, y_validation)
  print 'Classification Report:'
  predictions = grid_search.predict(X_val)
  print classification_report(y_validation, predictions)

def etapa2():
  names = [
    'MultinomialNB',
    'BernoulliNB',
    # 'KNeighborsClassifier',
    # 'RadiusNeighborsClassifier',
    # 'NuSVC',
    'RandomForestClassifier'
  ]

  clfs = [
    MultinomialNB(),
    BernoulliNB(),
    # KNeighborsClassifier(),
    # RadiusNeighborsClassifier(),
    # NuSVC(),
    RandomForestClassifier()
  ]

  param_grids = [
    json.load(open('parameters/Bayes/multinomial_bayes_params.json')),
    json.load(open('parameters/Bayes/bernoulli_bayes_params.json')),
    # json.load(open('parameters/Nearest Neighbors/knn_params.json')),
    # json.load(open('parameters/Nearest Neighbors/knn_radius_params.json')),
    # json.load(open('parameters/SVM/nusvc_params.json')),
    json.load(open('parameters/Trees/random_forest_params.json'))
  ]

  # El gaussian naive bayes no toma parametros, lo corro sin gridsearch
  clf = GaussianNB()
  scores = cross_val_score(clf, X.todense(), y_train, scoring='f1', cv=10)
  print 'GaussianNB'
  print np.mean(scores), np.std(scores)
  print 'Classification Report:'
  clf.fit(X.todense(), y_train)
  print 'Validation Score:', clf.score(X_val.todense(), y_validation)
  predictions = clf.predict(X_val.todense())
  print classification_report(y_validation, predictions)

  for i in range(0, len(param_grids)):
    grid_search = GridSearchCV(clfs[i], param_grid=param_grids[i], scoring='f1', cv=10, n_jobs=-1)
    grid_search.fit(X, y_train)
    print names[i]
    print 'Best Estimator:', str(grid_search.best_estimator_)
    print 'Best Score:', str(grid_search.best_score_)
    print 'Best Params:', str(grid_search.best_params_)
    print 'Validation Score:', grid_search.score(X_val, y_validation)
    print 'Classification Report:'
    predictions = grid_search.predict(X_val)
    print classification_report(y_validation, predictions)

def etapa3():
  names = [
    'DecisionTreeClasifier',
    'MultinomialNB',
    'BernoulliNB',
    'RandomForestClassifier',
    'GaussianNB'
  ]

  clfs = [
    DecisionTreeClassifier(max_features=None, splitter='best', criterion='entropy', max_depth=25),
    MultinomialNB(alpha=0.0, fit_prior=True),
    BernoulliNB(binarize=0.0, alpha=0.5, fit_prior=True),
    RandomForestClassifier(max_features='sqrt', n_estimators=15, bootstrap=False, criterion='entropy', max_depth=None)
  ]

  clf = GaussianNB()

  print 'Selection'
  # UNIVARIADA VARIANZA
  best_scores = [0.0 for _ in range(0, len(clfs)+1)]
  best_variances = [0.0 for _ in range(0, len(clfs)+1)]
  variances = [0.15, 0.2, 0.25]
  for var in variances:
    sel = VarianceThreshold(threshold=(var))
    X_new = sel.fit_transform(X.todense())
    X_val_new = sel.transform(X_val.todense())

    for i in range(0, len(clfs)):
      clfs[i].fit(X_new, y_train)
      score = clfs[i].score(X_val_new, y_validation)
      if(best_scores[i] < score):
        best_scores[i] = score
        best_variances[i] = var

    clf.fit(X_new, y_train)
    score = clf.score(X_val_new, y_validation)
    if(best_scores[len(clfs)] < score):
      best_scores[len(clfs)] = score
      best_variances[len(clfs)] = var

  for i in range(0, len(clfs)):
    print str(names[i])
    print 'Eliminando varianza menor a:', str(best_variances[i])
    print 'Con Score:', str(best_scores[i])

  print str(names[len(clfs)])
  print 'Eliminando varianza menor a:', str(best_variances[len(clfs)])
  print 'Con Score:', str(best_scores[len(clfs)])
  print

  # # UNIVARIADA PERCENTILES
  # best_scores = [0.0 for _ in range(0, len(clfs)+1)]
  # best_percentiles = [0.0 for _ in range(0, len(clfs)+1)]
  # percentiles = [5, 10, 20, 25, 50]
  # for per in percentiles:
  #   sel = SelectPercentile(score_func=f1_score, percentile=per)
  #   X_new = sel.fit_transform(X.todense(), y_train)
  #   X_val_new = sel.transform(X_val.todense())

  #   for i in range(0, len(clfs)):
  #     clfs[i].fit(X_new, y_train)
  #     score = clfs[i].score(X_val_new, y_validation)
  #     if(best_scores[i] < score):
  #       best_scores[i] = score
  #       best_percentiles[i] = per

  #   clf.fit(X_new, y_train)
  #   score = clf.score(X_val_new, y_validation)
  #   if(best_scores[len(clfs)] < score):
  #     best_scores[len(clfs)] = score
  #     best_percentiles[len(clfs)] = per

  # for i in range(0, len(clfs)):
  #   print str(names[i])
  #   print 'Usando percentil:', str(best_percentiles[i])
  #   print 'Con Score:', str(best_scores[i])

  # print str(names[len(clfs)])
  # print 'Usando percentil:', str(best_percentiles[len(clfs)])
  # print 'Con Score:', str(best_scores[len(clfs)])
  # print

  # print 'Transform'
  # # PCA
  # best_scores = [0.0 for _ in range(0, len(clfs)+1)]
  # best_variance_kept = [0.0 for _ in range(0, len(clfs)+1)]
  # variances_kept = [0.85, 0.9, 0.95, 0.99]
  # for var_kept in variances_kept:
  #   sel = PCA(n_components=var_kept)
  #   X_new = sel.fit_transform(X.todense())
  #   X_val_new = sel.transform(X_val.todense())

  #   for i in range(0, len(clfs)):
  #     clfs[i].fit(X_new, y_train)
  #     score = clfs[i].score(X_val_new, y_validation)
  #     if(best_scores[i] < score):
  #       best_scores[i] = score
  #       best_variance_kept[i] = var_kept

  #   clf.fit(X_new, y_train)
  #   score = clf.score(X_val_new, y_validation)
  #   if(best_scores[len(clfs)] < score):
  #     best_scores[len(clfs)] = score
  #     best_variance_kept[len(clfs)] = var_kept

  # for i in range(0, len(clfs)):
  #   print str(names[i])
  #   print 'Guardando porcentaje de varianza:', str(best_variance_kept[i])
  #   print 'Con Score:', str(best_scores[i])

  # print str(names[len(clfs)])
  # print 'Guardando porcentaje de varianza:', str(best_variance_kept[len(clfs)])
  # print 'Con Score:', str(best_scores[len(clfs)])

def modo_de_uso():
  print 'Modo de uso:'
  print 'Seleccionar etapa del TP que se quiere correr'
  print '1. Decision Tree Classifier'
  print '2. Naive Bayes, KNN, SVM, Random Forest'
  print '3. Reduccion de la dimensionalidad'

def main():
  if len(sys.argv) == 1:
    modo_de_uso()
    sys.exit()
  if not('1' in sys.argv or '2' in sys.argv or '3' in sys.argv):
    print 'Etapa no disponible'
  else:
    load_data()
    if '1' in sys.argv:
      etapa1()
    elif '2' in sys.argv:
      etapa2()
    else:
      etapa3()

if __name__ == '__main__':
  main()
