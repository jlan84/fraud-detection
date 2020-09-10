import pandas as pd
import cPickle as pickle
import psycopg2 as pg2
from config import config    # from the Postgres webpage (not sure what it does) 

def insert_event(event, prediction):
    '''Insert a new event into the event table'''

    sql = '''INSERT INTO table(event)
             VAlUES(%s) RETURNING event_id;'''
    conn = None
    vendor_id = None
    try: 
        # connect to database (not sure how this part works)
        params = config()
        conn = pg2.connect(**params)
        # create cursor
        cur = conn.cursor()
        # execute insert statement
        cur.execute(sql, (event, prediction))
        # get the generated id
        event_id = cur.fetchone()[0]
        # commit the changes to the database
        conn.commit()
        # close the communication with the database
        cur.close()
    except (Exception, pg2.DatabaseError) as error:
        print(error)
    finally:
            if conn is not None:
                conn.close
    return event_id

if __name__ == '__main__':
    # Read in single example
    event = pd.read_csv('data/test_script_examples.csv')

    # Vectorize example
    vc_event = vectorize_single(event)

    # Unpickle the model
    with open ('model.pkl') as f_un:
        model_unpickled = pickle.load(f_un)

    # Predict the label
    prediction = model_unpickled.predict_proba(event)

    # Output label probability
    print(prediction)

    # Add to Postgres/Mongo DB
    # ---Still need to create database, starting with simplest option I can
    # ---think of: one column for the event, one column for the prediction
    insert_event(event, prediction)