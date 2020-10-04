import requests
import json
import pickle
import pandas as pd
from flask import Flask, request, render_template
from mw_predict import vectorize_single, insert_event, text_from_html, return_all_events
app = Flask(__name__)  

#Threshold Testing
def fraud_warning_level(proba):
    if proba > 0.7:
        warning = '🚨 HIGH RISK 🚨'
    elif proba > 0.2:
        warning = '🚧 Medium Risk 🚧'
    else:
        warning = '✅ Low Risk ✅'
    return warning

def is_letter(self, letter):
        if letter.lower() in ['q','w','e','r','t','y','u','i','o','p','a','s',
                              'd','f','g','h','j','k','l','z','x','c','v','b',
                              'n','m']:
            return True
        else:
            return False

def is_capitalized(words):
        total = 0
        capitalized = 0
        words = str(words)
        for word in words:
            if is_letter(word) == 0:
                continue
            if word.isupper():
                capitalized += 1
            total +=1
        if total == 0:
            return 0
        else:
            prop_capitalized = capitalized/total
            return prop_capitalized

# Home Page
@app.route('/', methods=['GET', 'POST'])
def home():
    event = requests.get('http://galvanize-case-study-on-fraud.herokuapp.com/data_point').content 
    event_json = json.loads(event)
    event_df = pd.json_normalize(event_json)
    prediction = vectorize_single(event_df)
    event_name = event_json['name']
    event_desc = text_from_html(event_json['description'])
    event_payouts = event_json['previous_payouts']
    event_public_notice = (((event_json['event_start']
                            - event_json['event_published']) / 60) / 60) / 24
    event_private_notice = (((event_json['event_start']
                              - event_json['event_created']) / 60) / 60) / 24
    event_email = event_json['email_domain']
    event_country = event_json['country']
    event_caps_name = is_capitalized(event_json['name'])
    event_caps_desc = is_capitalized(text_from_html(event_json['description']))
    warn = fraud_warning_level(prediction[:, 1])
    insert_event(event_df.to_json(), prediction[0][1])
    return render_template('index.html', event_name=event_name,
                           event_desc=event_desc,
                           event_payouts=len(event_payouts),
                           event_public_notice=round(event_public_notice),
                           event_private_notice=round(event_private_notice),
                           event_email=event_email,
                           event_caps_desc=round(event_caps_desc * 100),
                           event_caps_name=round(event_caps_name * 100),
                           event_country=event_country, warning=warn)

# Blog Model Info
@app.route('/model', methods=['GET'])
def model():
    return render_template('blog.html')

@app.route('/database', methods=['GET'])
def database():
    table = return_all_events()
    return render_template('database.html', table=table)

@app.route('/hello', methods=['GET'])
def hello_world():
    return ''' <h1> Hello, World!</h1> '''

@app.route('/form_example', methods=['GET'])
def form_display():
    return ''' <form action="/string_reverse" method="POST">
                <input type="text" name="some_string" />
                <input type="submit" />
               </form>
             '''

@app.route('/string_reverse', methods=['POST'])
def reverse_string():
    text = str(request.form['some_string'])
    reversed_string = text[-1::-1]
    return ''' output: {}  '''.format(reversed_string)

@app.route('/fraud_warning', methods=['GET'])
def fraud_warning():
    warn_high = fraud_warning_level(0.80)
    warn_med = fraud_warning_level(0.52)
    warn_low = fraud_warning_level(0.20)
    return f'''<p> {warn_high} </p>
              <p> {warn_med} </p>
              <p> {warn_low} </p>
            '''

# Single Event Testing
@app.route('/event', methods=['GET'])
def event():
    event = requests.get('http://galvanize-case-study-on-fraud.herokuapp.com/data_point').content 
    event_json = json.loads(event)
    event_df = pd.json_normalize(event_json)
    prediction = vectorize_single(event_df)
    warn = fraud_warning_level(prediction[:, 1])
    event_id = insert_event(event_df.to_json(), prediction[0][1])
    table = return_all_events()
    return f'{table} FRAUD ANALYSIS: {warn} {prediction}'

if __name__ == '__main__':
    with open ('../data/model.pkl', 'rb') as f_un:
        model_unpickled = pickle.load(f_un) 

    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)