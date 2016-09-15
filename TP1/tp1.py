### http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter ###
import sys
import json
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

X_train = []
y_train = []
X_validation = []
y_validation = []

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

def etapa1():
  estimators = []
  scores = []
  params = []
  scorers = []
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

    # Extract max_features most frequent words from train folds
    vectorizer = CountVectorizer(token_pattern='[^\d\W_][\w|\']+', max_features=500)
    X = vectorizer.fit_transform(kf_X_train)
    # Apply the same extraction on test fold
    X_test = vectorizer.transform(kf_X_test)

    clf = DecisionTreeClassifier()
    param_grid = json.load(open('parameters/Trees/decision_tree_params.json'))
    grid_search = GridSearchCV(clf, param_grid=param_grid, scoring='f1')
    grid_search.fit(X, kf_y_train)
    estimators.append(grid_search.best_estimator_)
    scores.append(grid_search.best_score_)
    params.append(grid_search.best_params_)
    scorers.append(grid_search.scorer_)
  for i in range(0, 10):
    print str(i) + ' ' + str(estimators[i]) + ' ' + str(scores[i]) + ' ' + str(params[i]) + ' ' + str(scorers[i])

def etapa2():
  clfs = [
    GaussianNB(),
    MultinomialNB(),
    BernoulliNB(),
    KNeighborsClassifier(),
    RadiusNeighborsClassifier(),
    SVC(),
    NuSVC(),
    RandomForestClassifier()
  ]

  param_grids = [
    # Gaussian no toma parametros... habria que ver como se porta con el [], o poner un if bien cabeza en el for
    [],
    json.load(open('parameters/Bayes/multinomial_bayes_params.json')),
    json.load(open('parameters/Bayes/bernoulli_bayes_params.json')),
    json.load(open('parameters/Nearest Neighbors/knn_params.json')),
    json.load(open('parameters/Nearest Neighbors/knn_radius_params.json')),
    json.load(open('parameters/SVM/svc_params.json')),
    json.load(open('parameters/SVM/nusvc_params.json')),
    # Hay que cambiar este por random forest!
    json.load(open('parameters/Trees/decision_tree_params.json'))
  ]

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
