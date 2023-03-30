import pandas as pd
import random
import numpy as np

import pickle
from nltk.stem.porter import *
from sklearn.feature_extraction.text import CountVectorizer
    
import pandas as pd


def save_object(obj, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b



ANALYZER = CountVectorizer().build_analyzer()

STEMMER = PorterStemmer()

def stemmed_words(doc):
    return (STEMMER.stem(w) for w in ANALYZER(doc))



def get_bagOfWords(X, **args):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(**args)
    return pd.DataFrame(vectorizer.fit_transform(X).toarray(), index=X.index, columns=vectorizer.get_feature_names_out())


def get_tfidf(X, **args):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(**args)
    return pd.DataFrame(vectorizer.fit_transform(X).toarray(), index=X.index, columns=vectorizer.get_feature_names_out())


class Text2Features:
    def __init__(self, ngram_min = 1, ngram_max = 4, max_features = 100):
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.max_features = max_features
        
        analyzer = CountVectorizer().build_analyzer()
        
        self.count_vectorizer = CountVectorizer(analyzer=stemmed_words, ngram_range=(self.ngram_min,self.ngram_max), 
                                         max_features=self.max_features, 
                                         stop_words='english')
        
        self.tfidf_vectorizer = TfidfVectorizer(analyzer=stemmed_words, ngram_range=(self.ngram_min,self.ngram_max), 
                               max_features=self.max_features, 
                               stop_words='english')

        
    def fit(self, all_texts):
        self.count_vectorizer.fit(all_texts)
        self.tfidf_vectorizer.fit(all_texts)
        
        
    
    def transform(self, X):
        X_bagOfWords = pd.DataFrame(self.count_vectorizer.transform(X).toarray(), index=X.index, columns=self.count_vectorizer.get_feature_names_out())

        X_tfidf = pd.DataFrame(self.tfidf_vectorizer.transform(X).toarray(), index=X.index, columns=self.tfidf_vectorizer.get_feature_names_out())
        
        Xf = pd.concat([X_bagOfWords, X_tfidf], axis=1)
        
        return Xf
    
    
    
def make_model_fit(X, y, model_type = "NB", nb_use_prior = True):
    text2features = Text2Features(ngram_max=2, max_features=200)
    text2features.fit(X)
    Xt = text2features.transform(X)

    if model_type == "NB":
        from sklearn.naive_bayes import MultinomialNB
        clf = MultinomialNB(fit_prior=nb_use_prior)

    elif model_type == "SVM":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))

    elif model_type == "RF":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(max_depth=2, random_state=0)

    clf.fit(Xt, y)
    return text2features, clf

def make_test_dataframe(texts):
    df = pd.Series(texts)
    return df

def make_predictions(texts, feature_extractor, classifier_model):
    X_test = make_test_dataframe(texts)
    X_test = feature_extractor.transform(X_test)
    y_pred = classifier_model.predict_proba(X_test)
    return y_pred, classifier_model.classes_
    # preds_idx = np.argsort(-y_pred, axis = 1)[0]
    # predicted_classes = [classifier_model.classes_[i] for i in preds_idx][:15]
    # return predicted_classes
    


FILENAME = "relationship_phrases_new.pkl"


reldf = pd.read_csv("relations_labels.csv")
relations = list(set(list(reldf["uri"].values)))
relations_names = {rel: list(set(list(reldf[reldf["uri"] == rel]["label"].values))) for rel in relations}

relations_phrases = load_object(FILENAME)
relation_all_phrases = {key: relations_phrases[key[1:-1].split("/")[-1]] + relations_names[key] * 5 for key in relations_names.keys()}

relation_all_phrases_data = [[key.split("/")[-1][:-1], value[i]] for key, value in relation_all_phrases.items() for i in range(len(value))]

df = pd.DataFrame(relation_all_phrases_data)

X = df[1]

y = df[0]


feature_extractor, model = make_model_fit(X, y, "NB")

