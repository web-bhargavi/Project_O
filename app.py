from flask import Flask, render_template, request
from modules import module1
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/module1', methods=['GET', 'POST'])
def handle_module1():
    return module1.process(request, app.config['UPLOAD_FOLDER'])

if __name__ == '__main__':
    app.run(debug=True, port=5001)
