import cv2
import os
import base64
from flask import render_template
import numpy as np
from matplotlib import pyplot as plt

def convert_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def save_histogram(img, upload_folder):
    color = ('b', 'g', 'r')
    plt.figure()
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    hist_path = os.path.join(upload_folder, 'histogram.jpg')
    plt.savefig(hist_path)
    plt.close()
    return hist_path

def process(request, upload_folder):
    image_data = None
    hist_data = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(upload_folder, file.filename)
            file.save(filepath)

            img = cv2.imread(filepath)

            if 'segment' in request.form:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                lower_green = np.array([40, 40, 40])
                upper_green = np.array([70, 255, 255])
                mask = cv2.inRange(hsv, lower_green, upper_green)
                img = cv2.bitwise_and(img, img, mask=mask)

            if 'histogram' in request.form:
                hist_path = save_histogram(img, upload_folder)
                with open(hist_path, "rb") as f:
                    hist_data = base64.b64encode(f.read()).decode('utf-8')

            image_data = convert_to_base64(img)

    return render_template('module3.html', image_data=image_data, hist_data=hist_data)
