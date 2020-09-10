import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(13)

def initial_model(df):
    y = df.pop('fraud')
    X = df[['body_length', 'channels', 'delivery_method', 'event_created', 'event_end', 'event_published', 'event_start', 'has_logo', 'user_age', 'org_facebook', 'show_map']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    forest = RandomForestClassifier()

    forest.fit(X_train, y_train)
    preds = forest.predict(X_test)
    print(forest.score(X_test, y_test))
    print(f'F1 Score dumb model: {f1_score(y_test, preds, average="weighted")}')
    print(f'Presicion Score dumb model: {precision_score(y_test, preds, zero_division=0)}')
    print(confusion_matrix(y_test, preds))
    return forest

def split_email(email, get_ending=False):
    email = str(email)
    head, sep, tail = email.partition('.')
    if get_ending:
        return tail
    else:
        return head

if __name__ == "__main__":  
    df = pd.read_json('data/data.zip')
    # df = df.dropna()

    df['subdomain'] = df.apply(lambda row: split_email(row.email_domain), axis=1)
    df['tld'] = df.apply(lambda row: split_email(row.email_domain, get_ending=True), axis=1)

    # print(df.acct_type.unique())
    df['fraud'] = np.where((df['acct_type'] == 'fraudster') | (df['acct_type'] == 'fraudster_event') | (df['acct_type'] == 'fraudster_att'), 1, 0)

    print(split_email(df.email_domain.loc[[1]], get_ending=False))
    print(df.email_domain.loc[[1]])
    print(df.org_name.loc[[1]])
    print(df.loc[[48]].T)


    df['date_pub'] = pd.to_datetime(df.event_published, unit='s')
    df['date_start'] = pd.to_datetime(df.event_start, unit='s')

    df['public_notification_period'] = (((df.event_start - df.event_published) / 60) / 60) / 24
    df['private_notification_period'] = (((df.event_start - df.event_created) / 60) / 60) / 24

    mask = (df.fraud == 1)
    frauds = df[mask]

    # print(frauds.describe().T)
    # print(df.describe().T)


    # for i in range(50, 70):
    #     print(df.loc[[i]].T)
    #     for j in df.previous_payouts[i]:
    #             print(j)
    #     for j in df.ticket_types[i]:
    #             print(j)
    
    payout = df.payout_type
    fraud_i = df.fraud
    df2 = pd.DataFrame(payout, index=fraud_i)

    # print(df.head().T)
    # print(df.description[0])

    # # # forest = initial_model(df)

    # # list_cols = df.pop('previous_payouts')
    # tickets = df[['ticket_types', 'fraud']]
    # # print(list_cols[0])
    # # print(list_cols2[0])

    # for col in frauds.columns:
    #     print(col)
    #     print(len(frauds[col].unique()))
    #     if len(frauds[col].unique()) < 10:
    #         print(frauds[col].unique())
    #     print('======================================================================')
    # for email in df.email_domain.unique():
    #     print()