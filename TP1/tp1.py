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

from sklearn.decomposition import PCA

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
  vectorizer = CountVectorizer(token_pattern='[^\d\W_][\w|\']+', max_features=500)
  X = vectorizer.fit_transform(X_train)
  X_val = vectorizer.transform(X_validation)
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
    'KNeighborsClassifier',
    'RadiusNeighborsClassifier',
    'NuSVC',
    'RandomForestClassifier'
  ]

  clfs = [
    MultinomialNB(),
    BernoulliNB(),
    KNeighborsClassifier(),
    RadiusNeighborsClassifier(),
    NuSVC(),
    RandomForestClassifier()
  ]

  param_grids = [
    json.load(open('parameters/Bayes/multinomial_bayes_params.json')),
    json.load(open('parameters/Bayes/bernoulli_bayes_params.json')),
    json.load(open('parameters/Nearest Neighbors/knn_params.json')),
    json.load(open('parameters/Nearest Neighbors/knn_radius_params.json')),
    json.load(open('parameters/SVM/nusvc_params.json')),
    json.load(open('parameters/Trees/random_forest_params.json'))
  ]

  vectorizer = CountVectorizer(token_pattern='[^\d\W_][\w|\']+', max_features=500)
  X = vectorizer.fit_transform(X_train)
  X_val = vectorizer.transform(X_validation)

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
  pass
  #En PCA mandas PCA(n_components=None, copy=True, whiten=False)
  #N_COMPONENTS
  # n_components es básicamente cuántas componentes nos queremos quedar. El default se queda todas y si pones n_components='mle' dice esto
  		# if n_components == ‘mle’, Minka’s MLE is used to guess the dimension. 
  # Y si mandas un numero menor a 1, hace esto: select the number of components such that the amount of variance that needs to be explained is greater than the percentage specified by n_components
  # COPY
  # El default es TRUE
  # If False, data passed to fit are overwritten and running fit(X).transform(X) will not yield the expected results, use fit_transform(X) instead. (no entendi bien esto)
  # WHITEN
  # El default es FALSE
  # When True the components_ vectors are divided by n_samples times singular values to ensure uncorrelated outputs with unit component-wise variances. (Tampoco entendí esto, diría que no lo usemos)

  #Ejemplito

  #>>> import numpy as np
  #>>> from sklearn.decomposition import PCA
  #>>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
  #>>> pca = PCA(n_components=2)
  #>>> pca.fit(X)
  #PCA(copy=True, n_components=2, whiten=False)
  #>>> print(pca.explained_variance_ratio_) 
  #[ 0.99244...  0.00755...]

  #Metodos

  #fit_transform(X,y) --> Hace el fit y aplica la reducción de dimensionalidad
  #transform(X,y) ---> hace solo la reducción.
  #score(X,y) ---> Return the average log-likelihood of all samples
  #get_covariance()

  #Atributos

  #Le podes pedir las componentes
  # n_components_ te tira el numero de componentes (por si usaste el mle o 0 < n_componentes < 1)
  # explained_variance_ratio_ te tira el "Percentage of variance explained by each of the selected components"


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
