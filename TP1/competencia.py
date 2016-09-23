import sys
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

def run(file):
  ham_txt = json.load(open('dataset_dev/ham_dev.json'))
  spam_txt = json.load(open('dataset_dev/spam_dev.json'))
  new_txt = json.load(open(file))

  X_txt = ham_txt + spam_txt
  y = [0 for _ in range(len(ham_txt))] + [1 for _ in range(len(spam_txt))]

  vectorizer = CountVectorizer(token_pattern='[^\d\W_][\w|\']+', max_features=500)
  X = vectorizer.fit_transform(X_txt)
  X_test = vectorizer.transform(new_txt)

  clf = RandomForestClassifier(max_features='sqrt', n_estimators=15, bootstrap=False, criterion='entropy', max_depth=None)
  clf.fit(X, y)
  y_test = clf.predict(X_test)

  for pred in y_test:
    if(pred == 0):
      print 'ham'
    else:
      print 'spam'

def main():
  if len(sys.argv) == 1:
    print 'Requiere el path a un archivo'
    sys.exit()
  run(sys.argv[1])

if __name__ == '__main__':
  main()
