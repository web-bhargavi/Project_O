import cv2
import os
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import render_template

def convert_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def save_axis_image(img, upload_folder):
    h, w = img.shape[:2]
    x_step = max(w // 10, 50)
    y_step = max(h // 10, 50)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.xticks(np.arange(0, w, step=x_step))
    plt.yticks(np.arange(0, h, step=y_step))
    plt.grid(False)
    path = os.path.join(upload_folder, "axis_image.jpg")
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return path

def process(request, upload_folder):
    image_data = None
    axis_image_data = None
    msg = None
    image_size = None
    filename = None

    if request.method == 'POST':

        if 'image' in request.files:
            file = request.files['image']
            if file:
                filename = file.filename
                filepath = os.path.join(upload_folder, filename)
                file.save(filepath)
        else:
            filename = request.form.get('filename')

        if filename:
            filepath = os.path.join(upload_folder, filename)
            img = cv2.imread(filepath)

            axis_path = save_axis_image(img, upload_folder)
            with open(axis_path, "rb") as f:
                axis_image_data = base64.b64encode(f.read()).decode('utf-8')

            if request.form.get('show_size') == 'on':
                h, w = img.shape[:2]
                image_size = f"Width: {w}px, Height: {h}px"

            color_option = request.form.get('color')
            if color_option == 'gray':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif color_option == 'hsv':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            selected_channels = request.form.getlist('channels')
            if selected_channels:
                channel_images = []
                for ch in selected_channels:
                    if ch in ['R', 'G', 'B']:
                        c_index = {'B': 0, 'G': 1, 'R': 2}[ch]
                        single_channel = img[:, :, c_index]
                    elif ch in ['H', 'S', 'V']:
                        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                        c_index = {'H': 0, 'S': 1, 'V': 2}[ch]
                        single_channel = hsv[:, :, c_index]
                    else:
                        continue
                    single_channel_bgr = cv2.cvtColor(single_channel, cv2.COLOR_GRAY2BGR)
                    channel_images.append(single_channel_bgr)

                if channel_images:
                    try:
                        img = np.hstack(channel_images)
                    except:
                        img = channel_images[0]

            if request.form.get('resize') == 'on':
                width = int(request.form.get('width'))
                height = int(request.form.get('height'))
                img = cv2.resize(img, (width, height))

            if request.form.get('rotate') == 'on':
                angle = float(request.form.get('angle'))
                (h, w) = img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h))

            flip_option = request.form.get('flip')
            if flip_option == 'horizontal':
                img = cv2.flip(img, 1)
            elif flip_option == 'vertical':
                img = cv2.flip(img, 0)
            elif flip_option == 'both':
                img = cv2.flip(img, -1)

            if request.form.get('crop') == 'on':
                x1 = int(request.form.get('x1'))
                y1 = int(request.form.get('y1'))
                x2 = int(request.form.get('x2'))
                y2 = int(request.form.get('y2'))
                img = img[y1:y2, x1:x2]

            if request.form.get('annotate') == 'on':
                text = request.form.get('text')
                x = int(request.form.get('x'))
                y = int(request.form.get('y'))
                cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 2)

            if request.form.get('annotate_line') == 'on':
                x1 = int(request.form.get('lx1'))
                y1 = int(request.form.get('ly1'))
                x2 = int(request.form.get('lx2'))
                y2 = int(request.form.get('ly2'))
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if request.form.get('annotate_circle') == 'on':
                x = int(request.form.get('cx'))
                y = int(request.form.get('cy'))
                r = int(request.form.get('r'))
                cv2.circle(img, (x, y), r, (255, 0, 255), 2)

            if request.form.get('annotate_rectangle') == 'on':
                x1 = int(request.form.get('rx1'))
                y1 = int(request.form.get('ry1'))
                x2 = int(request.form.get('rx2'))
                y2 = int(request.form.get('ry2'))
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)

            if 'save' in request.form:
                save_path = os.path.join(upload_folder, "saved_" + filename)
                cv2.imwrite(save_path, img)
                msg = f"Image Saved Successfully as saved_{filename}"

            image_data = convert_to_base64(img)

    return render_template('module1.html',
                           image_data=image_data,
                           axis_image_data=axis_image_data,
                           msg=msg,
                           image_size=image_size,
                           filename=filename)
