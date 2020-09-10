import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score
import numpy as np
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz

np.random.seed(13)
plt.style.use('fivethirtyeight')

class aw_feature_eng:

    def __init__(self):
        pass


    def initial_model(self, df):
        df = df.dropna()

        y = df.pop('fraud')
        X = df[['body_length', 'delivery_method', 'event_created', 'event_end', 'event_published', 'event_start', 'has_logo', 'user_age', 'org_facebook', 'show_map']]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify= y, test_size=0.33)

        forest = RandomForestClassifier()

        forest.fit(X_train, y_train)
        preds = forest.predict(X_test)
        print(forest.score(X_test, y_test))
        print(f'F1 Score dumb model: {f1_score(y_test, preds, average="weighted")}')
        print(f'Presicion Score dumb model: {precision_score(y_test, preds, zero_division=0)}')
        print(confusion_matrix(y_test, preds))
        return forest

    def split_email(self, email, get_ending=False):
        email = str(email)
        head, sep, tail = email.partition('.')
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

    def feature_eng(self, df):
        # Insert our target of fraud = 1 and not fraud = 0
        df['fraud'] = np.where((df['acct_type'] == 'fraudster') | (df['acct_type'] == 'fraudster_event') | (df['acct_type'] == 'fraudster_att'), 1, 0)

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

        #Return the feature engineered DataFrame
        return df

if __name__ == "__main__":  
    feat_eng = aw_feature_eng()
    df = feat_eng.feature_eng(pd.read_json('data/data.zip'))


    print(feat_eng.split_email(df.email_domain.loc[[1]], get_ending=False))
    print(df.email_domain.loc[[1]])
    print(df.org_name.loc[[1]])
    print(df.loc[[48]].T)
    print(df.delivery_method.unique())

    fraud, not_fraud = feat_eng.split_on_fraud(df, describe=True)

    fraud.org_subdomain_similarity.hist(bins=40)
    plt.show()
    not_fraud.org_subdomain_similarity.hist(bins=40)
    plt.show()

    # forest = initial_model(df)
