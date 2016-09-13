import json
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# UNCOMMENT TU PROPIA AVENTURA

#### ------ TREES
## DECISION TREE
#estimator=DecisionTreeClassifier()
#params=json.load(open('decision_tree_params.json'))
#
#### ------ BAYES
## GAUSSIAN BAYES
## http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB
#estimator=GaussianNB()
#params=None
#
## MULTINOMIAL BAYES
## http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB
#estimator=MultinomialNB()
#params=json.load(open('parameters/Bayes/multinomial_bayes_params.json'))
#
## BERNOULLI BAYES
## http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB
#estimator=BernoulliNB()
#params=json.load(open('parameters/Bayes/bernoulli_bayes_params.json'))
#
#### ------ NEAREST NEIGHBOURS
## KNN
## http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
#estimator=KNeighborsClassifier()
#params=json.load(open('parameters/Nearest Neighbors/knn_params.json'))
#
## KNN-Radius
## http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html#sklearn.neighbors.RadiusNeighborsClassifier
#estimator=RadiusNeighborsClassifier()
#params=json.load(open('parameters/Nearest Neighbors/knn_radius_params.json'))
#
#### ------ SVM
## SVC
## http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
#estimator=SVC()
#params=json.load(open('parameters/SVM/svc_params.json'))
#
## LinearSVC
## http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
#estimator=LinearSVC()
#params=json.load(open('parameters/SVM/linearsvc_params.json'))
#
## NuSVC
## http://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html#sklearn.svm.NuSVC
#estimator=NuSVC()
#params=json.load(open('parameters/SVM/nusvc_params.json'))
#
#### ------ TREES
## DECISION TREE
## http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
#estimator=DecisionTreeClassifier()
#params=json.load(open('parameters/Trees/decision_tree_params.json'))


grid=GridSearchCV(estimator, param_grid=params)

grid.fit(X,y)
grid.best_score_
grid.best_params_