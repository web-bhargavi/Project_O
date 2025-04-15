import cv2
import os
import base64
from flask import render_template

def convert_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def process(request, upload_folder):
    image_data = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(upload_folder, file.filename)
            file.save(filepath)

            img = cv2.imread(filepath)

            if 'add' in request.form:
                img = cv2.add(img, (50, 50, 50, 0))

            if 'invert' in request.form:
                img = cv2.bitwise_not(img)

            if 'threshold' in request.form:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            image_data = convert_to_base64(img)

    return render_template('module2.html', image_data=image_data)
