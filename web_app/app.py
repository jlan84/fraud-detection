import requests
import json
from flask import Flask, request, render_template
app = Flask(__name__)  

#Threshold Testing
def fraud_warning_level(proba):
    if proba > 0.75:
        warning = '🚨FRAUD🚨'
    elif proba > 0.5:
        warning = '🚧 Possible Fraud'
    else:
        warning = 'Not Fraud'
    return warning

# Home Page
@app.route('/', methods=['GET'])
def home():
    event = requests.get('http://galvanize-case-study-on-fraud.herokuapp.com/data_point').content 
    event_json = json.loads(event)
    event_name = event_json['name']
    warn = fraud_warning_level(0.52)
    return render_template('index.html', event_name=event_name, warning=warn)

@app.route('/model', methods=['GET'])
def model():
    return render_template('blog.html')

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
    warn = fraud_warning_level(0.52)
    return warn

@app.route('/event', methods=['GET'])
def event():
    event = requests.get('http://galvanize-case-study-on-fraud.herokuapp.com/data_point').content 
    event_json = json.loads(event)
    warn = fraud_warning_level(0.52)
    return f'{event_json["name"]} FRAUD ALERT: {warn}'

if __name__ == '__main__':
    # with open('../data/model.pkl') as f_un:
    #     model_unpickled = pickle.load(f_un)  

    app.run(host='0.0.0.0', port=8080, debug=True)