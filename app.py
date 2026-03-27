from flask import Flask, request, render_template
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

print("Starting app...")

try:
    print("Loading model...")
    model = pickle.load(open('model.pkl', 'rb'))

    print("Loading scaler...")
    scaler = pickle.load(open('scaler.pkl', 'rb'))

    print("Loading columns...")
    columns = pickle.load(open('columns.pkl', 'rb'))

    print("All files loaded successfully!")

except Exception as e:
    print("ERROR LOADING FILES:", e)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form.to_dict()

    df = pd.DataFrame([input_data])
    df = df.apply(pd.to_numeric)

    prediction = model.predict(df)

    return render_template(
        'index.html',
        prediction_text=f"Estimated Price: ${prediction[0]:,.2f}",
        input_data=input_data   
    )


if __name__ == "__main__":
    app.run(debug=True)