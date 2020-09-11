import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score
import numpy as np
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
from bs4 import BeautifulSoup
from bs4.element import Comment
from nltk.corpus import stopwords
import pickle as pickle

from justin_scripts import NaiveBayes

np.random.seed(42)
plt.style.use('fivethirtyeight')

class aw_feature_eng:

    def __init__(self):
        pass

    def create_nb_col(self, df):

        stop_words = stopwords.words('english')
        extra_stops = ['tickets', '00']
        stop_words.extend(extra_stops)
        df['desc_text'] = df.apply(lambda row: self.text_from_html(row['description']), axis=1)
        df['fraud'] =  np.where((df['acct_type'] == 'fraudster') | (df['acct_type'] == 'fraudster_event') | (df['acct_type'] == 'fraudster_att'), 1, 0)

        nb = NaiveBayes(df, stop_words)

        return nb

    def process_one(self, desc, stop_words):
        desc_text = str(self.text_from_html(desc))
        lower = desc_text.lower()
        split = lower.split()
        stops = [w for w in split if w not in stop_words]
        stop_num = [w for w in stops if not (w.isdigit() 
                        or w[0] == '-' and w[1:].isdigit())]
        X_ = ' '.join(stop_num)
        X_ = [X_]
        # cv = CountVectorizer()
        # count_mx = cv.fit_transform(X)
        # tfidf_mx = tfidf_matrix.transform(count_mx)
        return X_



    def initial_model(self, df, means=True):
        print(df.head().T)

        y = np.where((df['acct_type'] == 'fraudster') | (df['acct_type'] == 'fraudster_event') | (df['acct_type'] == 'fraudster_att'), 1, 0)
            
        stop_words = stopwords.words('english')
        extra_stops = ['tickets', '00']
        stop_words.extend(extra_stops)
        # nb = self.create_nb_col(df)
        # nb.generate_train_test()
        # nb.remove_X_train_stops()
        # nb.remove_X_test_stops()
        # nb.tf_idf_matrix()
        # nb.naive_bayes_model()
        # nb.return_top_n_words()

        # nb.get_accuracy_classification_report()

        with open('nb_pipeline.pkl', 'rb') as f:
            nb_pipeline = pickle.load(f)

        df['fraud'] = np.where((df['acct_type'] == 'fraudster') | (df['acct_type'] == 'fraudster_event') | (df['acct_type'] == 'fraudster_att'), 1, 0)

        

        df['nb_proba'] = df.apply(lambda row: nb_pipeline.predict_proba(self.process_one(row.description, stop_words))[0][1], axis=1)

        # print(df.head().T)

        drop_list = ['venue_latitude', 'venue_longitude','has_header', 'fraud', 'acct_type', 'approx_payout_date', 'event_end', 'event_start', 'gts', 'num_payouts', 'payout_type', 'sale_duration', 'sale_duration2', 'ticket_types', 'num_order']
        df = df.drop(drop_list, axis=1)

        X = df._get_numeric_data()

        print(X.describe().T)

        if means:
            X = X.fillna(X.mean())
        else:
            X = X.fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify= y, test_size=0.33)

        forest = RandomForestClassifier()

        forest.fit(X_train, y_train)
        preds = forest.predict(X_test)
        print(forest.score(X_test, y_test))
        print(f'F1 Score model: {f1_score(y_test, preds)}')
        print(f'Presicion Score dumb model: {precision_score(y_test, preds, zero_division=0)}')
        print(confusion_matrix(y_test, preds))
        return forest, y_test, X_test

    def split_email(self, email, get_ending=False):
        email = str(email)
        head, _, tail = email.partition('.')
        if get_ending:
            return tail
        else:
            return head

    def split_on_fraud(self, df, describe=False):
        mask = (df.fraud == 1)
        frauds = df[mask]
        not_fraud = df[~mask]
        if describe:
            print('~~~~~~~~~~~~~~~~~~~ FRAUD ~~~~~~~~~~~~~~~~~~~')
            print(frauds.describe().T)
            print('~~~~~~~~~~~~~~~~~~~ REAL ~~~~~~~~~~~~~~~~~~~')
            print(not_fraud.describe().T)
        return frauds, not_fraud

    def is_letter(self, letter):
        if letter.lower() in ['q','w','e','r','t','y','u','i','o','p','a','s','d','f','g','h','j','k','l','z','x','c','v','b','n','m']:
            return True
        else:
            return False


    def is_capitalized(self, words):
        total = 0
        capitalized = 0
        words = str(words)
        for word in words:
            if self.is_letter(word) == 0:
                continue
            if word.isupper():
                capitalized += 1
            total +=1
        if total == 0:
            return 0
        else:
            prop_capitalized = capitalized/total
            return prop_capitalized

    def feature_eng(self, df):
        # Insert our target of fraud = 1 and not fraud = 0
        #df['fraud'] = np.where((df['acct_type'] == 'fraudster') | (df['acct_type'] == 'fraudster_event') | (df['acct_type'] == 'fraudster_att'), 1, 0)
        
        #break down the email domain into two categories subdomain and tld
        df['subdomain'] = df.apply(lambda row: self.split_email(row.email_domain), axis=1)
        df['tld'] = df.apply(lambda row: self.split_email(row.email_domain, get_ending=True), axis=1)


        # Use fuzzywuzzy to find the similarity of the email's subdomain and the organization's name
        df['org_subdomain_similarity'] = df.apply(lambda row: fuzz.token_set_ratio(str(row.subdomain), str(row.org_name)), axis=1)

        #Count the total number of previous payouts to the user
        df['num_previous'] = df.apply(lambda row: len(row.previous_payouts), axis=1)

        #convert columns from seconds to DateTime
        df['date_pub'] = pd.to_datetime(df.event_published, unit='s')
        df['date_start'] = pd.to_datetime(df.event_start, unit='s')

        #Find the number of days until the event from both when it was published and when it was created
        df['public_notification_period'] = (((df.event_start - df.event_published) / 60) / 60) / 24
        df['private_notification_period'] = (((df.event_start - df.event_created) / 60) / 60) / 24


        df['capitalized'] = df.apply(lambda row: self.is_capitalized(row['name']), axis=1)

        df['desc_cap'] = df.apply(lambda row: self.is_capitalized(self.text_from_html(row['description'])), axis=1)

        #df['random_baseline'] = df.apply(lambda row: np.random.rand(), axis=1)

        #Return the feature engineered DataFrame
        return df

    def tag_visible(self, element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        if isinstance(element, Comment):
            return False
        return True

    def text_from_html(self, body):
        soup = BeautifulSoup(body, 'html.parser')
        texts = soup.findAll(text=True)
        visible_texts = filter(self.tag_visible, texts)  
        return u" ".join(t.strip() for t in visible_texts)

def conf_mat(preds, y_test):
    tn = 0
    tp = 0
    fn = 0
    fp = 0
    for pred, test in zip(preds, y_test):
        if test:
            if pred:
                tp += 1
            else:
                fn += 1
        else:
            if pred:
                fp += 1
            else:
                tn += 1

    return tn, fp, fn, tp

if __name__ == "__main__":  
    feat_eng = aw_feature_eng()
    # df = feat_eng.feature_eng(pd.read_json('data/data.zip'))
    # print(feat_eng.is_capitalized(df.name[0]))
    # print(df.info())
    # # df['desc_text'] = df.apply(lambda row: text_from_html(row.description), axis=1)


    # print(feat_eng.split_email(df.email_domain.loc[[1]], get_ending=False))
    # print(df.email_domain.loc[[1]])
    # print(df.org_name.loc[[1]])
    # print(df.loc[[48]].T)
    # print(df.delivery_method.unique())

    # # fraud, not_fraud = feat_eng.split_on_fraud(df, describe=True)

    # # fraud.org_subdomain_similarity.hist(bins=40)
    # # plt.show()
    # # not_fraud.org_subdomain_similarity.hist(bins=40)
    # # plt.show()

    # print(df.tld.unique())

    # forest = feat_eng.initial_model(df)
    rf, y_test, X_test = feat_eng.initial_model(feat_eng.feature_eng(pd.read_json('../data/data.zip')), means=False)

    # with open('rf_pipeline.pkl', 'wb') as f:
    #             pickle.dump(rf,f)

    thresholds = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    preds = rf.predict_proba(X_test)

    for threshold in thresholds:
        predicted = rf.predict_proba(X_test)
        new_predicted = []
        # new_predicted[:,0] = (predicted[:,0] < threshold).astype('int')
        new_predicted = (predicted[:,1] >= threshold).astype('int')
        print(predicted.shape)
        print(f"Thresh:{threshold}   tn, fp, fn, tp : {conf_mat(new_predicted, y_test)}")



    important_features_dict = {}
    for x,i in enumerate(rf.feature_importances_):
        important_features_dict[x]=i


    important_features_list = sorted(important_features_dict,
                                    key=important_features_dict.get,
                                    reverse=True)

    print('Most important features: %s' %important_features_list)

    for i in important_features_list:
        print(f' {i}: {important_features_dict[i]}')

