import pandas as pd
import pickle
import psycopg2 as pg2
# from config import config    # from the Postgres webpage (not sure what it does) 

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

def vectorize_single(event):
    # drop_list = ['acct_type', 'approx_payout_date', 'event_end', 'event_start',
    #              'gts', 'num_payouts', 'payout_type', 'sale_duration',
    #              'sale_duration2','ticket_types']
    keep_list = ['body_length', 'channels', 'delivery_method', 'event_created',
       'event_published', 'fb_published', 'has_analytics', 'has_header',
       'has_logo', 'name_length', 'num_order', 'object_id', 'org_facebook',
       'org_twitter', 'show_map', 'user_age', 'user_created', 'user_type',
       'venue_latitude', 'venue_longitude']
    event_dropped = event[keep_list]
    event_filled = event_dropped.fillna(0)
    event_for_model = event_filled._get_numeric_data()
    return event_for_model

if __name__ == '__main__':
    # Read in single example
    event = pd.read_csv('data/test_script_examples.csv', nrows=1, index_col='Unnamed: 0')

    # Vectorize example
    vc_event = vectorize_single(event)
    print(vc_event.columns)

    # Unpickle the model
    with open ('data/model.pkl', 'rb') as f_un:
        model_unpickled = pickle.load(f_un)

    # Predict the label
    prediction = model_unpickled.predict_proba(vc_event)

    # Output label probability
    print(prediction)

    # Add to Postgres/Mongo DB
    # ---Still need to create database, starting with simplest option I can
    # ---think of: one column for the event, one column for the prediction
    # insert_event(event, prediction)