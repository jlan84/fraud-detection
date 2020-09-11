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

class pipeliner():

    def __init__(self):
        with open('rf_pipeline.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        self.model = rf_model

    def pipeline(self, df):

        with open('nb_pipeline.pkl', 'rb') as f:
            nb_pipeline = pickle.load(f)

           
        stop_words = stopwords.words('english')
        extra_stops = ['tickets', '00']
        stop_words.extend(extra_stops)

        df['nb_proba'] = df.apply(lambda row: nb_pipeline.predict_proba(self.process_one(row.description, stop_words))[0][0], axis=1)

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


        drop_list = ['venue_latitude', 'venue_longitude','has_header','approx_payout_date', 'event_end', 'event_start', 'gts', 'num_payouts', 'payout_type', 'sale_duration', 'sale_duration2', 'ticket_types', 'num_order']
        df = df.drop(drop_list, axis=1)

        X = df._get_numeric_data()

        return X.fillna(0)



    def predict_fraud(self, data):
        piped_data = self.pipeline(data)
        return self.model.predict_proba(piped_data)

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

    def split_email(self, email, get_ending=False):
        email = str(email)
        head, _, tail = email.partition('.')
        if get_ending:
            return tail
        else:
            return head

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


if __name__ == "__main__":
    df = pd.read_json('../data/data.zip')
    pipe = pipeliner()
    n=0
    print(pipe.predict_fraud(df.loc[[n]]))
    print(df.loc[[n]].T)
