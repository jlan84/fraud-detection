import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score
import numpy as np

np.random.seed(13)

if __name__ == "__main__":  
    df = pd.read_json('data/data.zip')
    df = df.dropna()


    # mask = (df.user_age < 5)
    # gmail = df[mask]
    # print(gmail.head(25))

    print(df.acct_type.unique())
    
    y = df.pop('acct_type')
    df.country = df.country.astype('category')
    df.currency = df.currency.astype('category')
    _ = df.pop('description')
    X = df[['body_length', 'channels', 'delivery_method', 'event_created', 'event_end', 'event_published', 'event_start', 'has_logo', 'user_age', 'org_facebook', 'show_map']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    forest = RandomForestClassifier()

    forest.fit(X_train, y_train)
    preds = forest.predict(X_test)
    print(forest.score(X_test, y_test))
    print(f'F1 Score dumb model: {f1_score(y_test, preds, average="weighted")}')
    print(f'Presicion Score dumb model: {precision_score(y_test, preds, average="weighted", zero_division=0)}')