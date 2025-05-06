from flask import Flask, render_template, request
from modules import module1
from modules import module2
from modules import module3
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

@app.route('/module2', methods=['GET', 'POST'])
def handle_module2():
    return module2.process(request, app.config['UPLOAD_FOLDER'])

@app.route('/module3', methods=['GET', 'POST'])
def handle_module3():
    return module3.process(request, app.config['UPLOAD_FOLDER'])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
