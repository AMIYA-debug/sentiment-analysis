import os
from flask import Flask, render_template, request

app = Flask(__name__)


from notebook_funcs import load_resources, predict_review


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    review = request.form.get('review')
    if not review:
        return render_template('home.html', review=review, label=None, prob=None, error='Please enter a review')
    label, prob = predict_review(review)
    return render_template('home.html', review=review, label=label, prob=prob, error=None)


if __name__ == '__main__':
    # load heavy resources then run
    load_resources()
    app.run(host='0.0.0.0', port=5000, debug=True)
