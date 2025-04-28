from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
cv = pickle.load(open('models/cv.pkl', 'rb'))
clf = pickle.load(open('models/clf.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email = request.form.get('email-content')
    tokenized_mail = cv.transform([email])
    prediction = clf.predict(tokenized_mail)
    return render_template('index.html', prediction=prediction, email=email)

if __name__=='__main__':
    app.run(port='8080', debug=True)