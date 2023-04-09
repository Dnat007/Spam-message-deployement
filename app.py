from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained model and vectorizer
model = pickle.load(open('MNB.pkl', 'rb'))

vectorizer = pickle.load(open('tf_idf.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = vectorizer.transform(data).toarray()
        my_prediction = model.predict(vect)
        return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run()
