from flask import Flask, render_template, request
from inference import get_category

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img = request.files['file']
        categories, pred = get_category(img)
        return render_template('index.html', categories=categories, pred=pred)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='localhost', port=8000, debug=True)