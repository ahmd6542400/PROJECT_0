from flask import Flask,render_template,request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def homepage():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text=f'Mark is equal: {output}')

if __name__ == '__main__':
    app.run(debug=True,port=9000)