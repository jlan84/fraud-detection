import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
from bs4.element import Comment
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import re
import matplotlib.pyplot as plt
import pickle as pickle


plt.style.use('fivethirtyeight')
df = pd.read_json('../data/data.json')

def tag_visible(element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', 
                                   '[document]']:
            return False
        if isinstance(element, Comment):
            return False
        return True

def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)

def process_one(df, stop_words):
        df['desc_text'] = text_from_html(df.loc['description'])
        lower = df['desc_text'].lower()
        split = lower.split()
        stops = [w for w in split if w not in stop_words]
        stop_num = [w for w in stops if not (w.isdigit() 
                        or w[0] == '-' and w[1:].isdigit())]
        X = ' '.join(stop_num)
        X = [X]
        # cv = CountVectorizer()
        # count_mx = cv.fit_transform(X)
        # tfidf_mx = tfidf_matrix.transform(count_mx)
        return X

class NaiveBayes():

    def __init__(self, df, stop_words):
        self.df = df
        self.train_cv = None
        self.train_count_matrix = None
        self.train_tfidf_matrix = None
        self.nb_model = None
        self.stop_words = stop_words

    def description_to_text(self):
        self.df['desc_text'] = 0
        for i in range(self.df.shape[0]):
            self.df.loc[i,'desc_text'] = text_from_html(self.df.loc[i,'description'])
        
    def sub_set_df(self):
        df_fraud = self.df[self.df['fraud'] == 1]
        df_nf = self.df[self.df['fraud'] == 0]
        df_nf_subset = df_nf.sample(n=df_fraud.shape[0], replace=False, random_state=1720)
        self.df_combined = df_fraud.append(df_nf_subset, ignore_index=True)

    def generate_train_test(self, train_size=0.75):
        self.y = self.df_combined.pop('fraud').values
        self.X = self.df_combined['desc_text'].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                                                                self.X, self.y, 
                                                                stratify=self.y, 
                                                                train_size=train_size,
                                                                random_state = 123)
    def join_list_of_strings(self, lst):
        """
        Joins the list into a string
        
        Params
        lst: list of words
        """
        return [" ".join(x) for x in lst]

    def remove_X_train_stops(self):
        lower = [x.lower() for x in self.X_train]
        split_lst = [txt.split() for txt in lower]
        self.stops_removed_lst = []

        for split in split_lst:
            stops = [w for w in split if w not in self.stop_words]
            stop_num = [w for w in stops if not (w.isdigit() 
                        or w[0] == '-' and w[1:].isdigit())]
            self.stops_removed_lst.append(stop_num)
        self.X_train = self.join_list_of_strings(self.stops_removed_lst)

    def remove_X_test_stops(self):
        lower = [x.lower() for x in self.X_test]
        split_lst = [txt.split() for txt in lower]
        self.stops_removed_lst = []

        for split in split_lst:
            stops = [w for w in split if w not in self.stop_words]
            stop_num = [w for w in stops if not (w.isdigit() 
                        or w[0] == '-' and w[1:].isdigit())]
            self.stops_removed_lst.append(stop_num)
        self.X_test = self.join_list_of_strings(self.stops_removed_lst)
    
    def tf_idf_matrix(self):
            """
            Sets up a word count matrix, a tfidf matrix, and a CountVectorizer for
            the documents in the directory

            Params
            documents: list of strings to be vectorized

            Returns
            count_matrix: matrix with word counts
            tfidf_matrix: a tfidf matrix of the documents
            cv: CountVectorizer object for the documents
            """
            self.train_cv = CountVectorizer()
            self.train_count_matrix = self.train_cv.fit_transform(self.X_train)
            self.tfidf_transformer = TfidfTransformer()
            self.train_tfidf_matrix = self.tfidf_transformer.fit_transform(self.train_count_matrix)

    def naive_bayes_model(self):
            """
            Sets up a naive bayes model for the documents in the directory

            Params
            directory: directory for the documents
            stop_words: list of stop_words for word filtration
            technique: technique: str choose from ['porter', 'snowball','wordnet']

            Returns
            nb_model: a naive bayes model for the documents in the directory
            cv: CountVectorizer object for the documents
            """
            self.nb_model = MultinomialNB()
            self.nb_model.fit(self.train_tfidf_matrix, self.y_train)

    def pickle_model(self):
            self.description_to_text()
            self.sub_set_df()
            self.generate_train_test()
            self.remove_X_train_stops()
            self.remove_X_test_stops()
            self.tf_idf_matrix()
            nb = MultinomialNB()
            nb.fit(self.train_tfidf_matrix, self.y_train)
            with open('bayes.pkl', 'wb') as f:
                pickle.dump(nb,f)
            with open('tfidf.pkl', 'wb') as f:
                pickle.dump(self.tfidf_transformer ,f)

    def return_top_n_words(self, n=7):
            """
            Prints out the top n words for each document in the categories for the 
            documents in the directory

            Params
            directory: directory for the documents
            stop_words: list of stop_words for word filtration
            documents: a list of the categories (folders) in the directory
            technique: technique: str choose from ['porter', 'snowball','wordnet']

            """
            feature_words = self.train_cv.get_feature_names()
            categories = self.nb_model.classes_
            self.top_words_dic = {}
            for cat in range(len(categories)):
                print(f"\n Target: {cat}, name: {categories[cat]}")
                log_prob = self.nb_model.feature_log_prob_[cat]
                i_topn = np.argsort(log_prob)[::-1][:n]
                features_topn = [feature_words[i] for i in i_topn]
                self.top_words_dic[categories[cat]] = features_topn
                print(f"Top {n} tokens: ", features_topn)
    
    def get_accuracy_classification_report(self):
        """
        Prints out and returns the accuracy score from the prediction vs the actuals
        for the test set

        Params
        train_docs: list of strs used to train on
        test_docs: list of strs used to test
        test_targes: list of strs for the test target values

        Returns
        Accuracy score for the model
        """
        self.nb_pipeline = Pipeline([('vect', CountVectorizer()),
                            ('tfidf', TfidfTransformer()),
                            ('model', MultinomialNB()),
                            ])

        self.nb_pipeline.fit(self.X_train, self.y_train)
        with open('nb_pipeline.pkl', 'wb') as f:
            pickle.dump(self.nb_pipeline, f)
        self.predicted = self.nb_pipeline.predict(self.X_test)
        self.accuracy = np.mean(self.predicted == self.y_test)
        print("\nThe accuracy on the test set is {0:0.3f}.".format(self.accuracy))
        self.class_report = classification_report(self.y_test, self.predicted, 
                                                  digits=3,output_dict=True)

    def confustion_matrix_plot(self, ax):
        """
        Generates a confusion matrix

        Params
        test_docs: list of strings from the test set
        test_targets: list test target strings associated with the test_docs
        ax: axes to be used for the plot
        """
        plot_confusion_matrix(self.nb_pipeline, self.X_test, self.y_test, xticks_rotation='vertical',
                              cmap=plt.cm.Blues,values_format='d', ax=ax)
    
    def predic_probablility(self, X):
        return self.nb_pipeline.predict_proba(X)

if __name__ == "__main__":


    df['fraud'] = np.where((df['acct_type'] == 'fraudster') | 
                        (df['acct_type'] == 'fraudster_event') |
                        (df['acct_type'] == 'fraudster_att'), 1, 0)
    stop_words = stopwords.words('english')
    extra_stops = ['tickets', 00, 'event']
    stop_words.extend(extra_stops)

    # with open('nb_pipeline.pkl', 'rb') as f:
    #     nb_pipeline = pickle.load(f)
    
    # new_df = df.iloc[1,:]
    # X = process_one(new_df, stop_words)
    # print(nb_pipeline.predict_proba(X))
    nb = NaiveBayes(df, stop_words)
    nb.description_to_text()
    nb.sub_set_df()
    nb.generate_train_test()
    nb.remove_X_train_stops()
    nb.remove_X_test_stops()
    
    nb.tf_idf_matrix()
    nb.naive_bayes_model()
    nb.return_top_n_words()
    nb.get_accuracy_classification_report()
    # print(nb.class_report)
    # # probs = nb.predic_probablility(df['description'])
    # # print(type(probs))
    # # print(probs.shape)
    # # print(p)

    fig, ax = plt.subplots(figsize=(12,12))
    nb.confustion_matrix_plot(ax)
    ax.set_title('Fraud Confusion')
    plt.tight_layout()
    plt.show()
