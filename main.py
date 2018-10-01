

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

train = fetch_20newsgroups(subset='train', shuffle=True, random_state=7)

train.data[1]
train.data[7]
train.target_names
train.target[:len(train.target_names)]


print("\n".join(train.data[0].split("\n")[:7]))


count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train.data)
X_train_counts.shape # (#n_samples, #n_features)


len(count_vect.get_feature_names())
count_vect.get_feature_names()

count_vect.vocabulary
count_vect.vocabulary_.get(u'simm')


#Term Frequency times inverse document frequency.
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

len(X_train_tfidf[0].data) #<1x130107 sparse matrix of type '<class 'numpy.float64'>' with 69 stored elements in Compressed Sparse Row format>
X_train_tfidf[0].data[:7]



clf = MultinomialNB().fit(X_train_tfidf, train.target)


# less code 
# the previous part can be reduce by using Pipeline() 
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_clf = text_clf.fit(train.data, train.target)

def accur(a,b):
    return np.mean(a == b)


# NB Classifier:
test = fetch_20newsgroups(subset='test', shuffle=True, random_state=7)
predicted = text_clf.predict(test.data)

print("The accuracy is : " + str(accur(predicted,test.target)*100)+"%")


# SVM Classifier:
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf-svm', SGDClassifier(alpha=1e-3, max_iter=5, random_state=7))])
text_clf_svm = text_clf_svm.fit(train.data, train.target)
predicted_svm = text_clf_svm.predict(test.data)
print("The accuracy is : " + str(accur(predicted_svm,test.target)*100)+"%")


# GridSearchCV Classifier Tunning:
parameters = {'vect__ngram_range': [(1, 2), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-3, 1e-4)}
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(train.data, train.target)

gs_clf.best_score_
gs_clf.best_params_


parameters_svm = {'vect__ngram_range': [(1, 2), (1, 2)], 'tfidf__use_idf': (True, False),'clf-svm__alpha': (1e-3, 1e-4)}

gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(train.data, train.target)


gs_clf_svm.best_score_
gs_clf_svm.best_params_









