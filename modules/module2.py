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
    path = os.path.join(upload_folder, "module2_axis_image.jpg")
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

            # === Arithmetic Operations ===
            try:
                if request.form.get('add_brightness') == 'on':
                    val = int(request.form.get('add_value', 50))
                    img = cv2.add(img, val)

                if request.form.get('sub_brightness') == 'on':
                    val = int(request.form.get('sub_value', 50))
                    img = cv2.subtract(img, val)

                if request.form.get('contrast_multiply') == 'on':
                    val = float(request.form.get('mult_value', 1.2))
                    img = cv2.convertScaleAbs(img, alpha=val, beta=0)

                if request.form.get('contrast_divide') == 'on':
                    val = float(request.form.get('div_value', 1.2))
                    img = cv2.convertScaleAbs(img, alpha=1.0 / val, beta=0)
            except Exception as e:
                msg = f"Error in arithmetic operation: {str(e)}"
                return render_template('module2.html', msg=msg, axis_image_data=axis_image_data, filename=filename)

            # === Thresholding ===
            try:
                thresh_type = request.form.get('thresh_type', 'none')
                if thresh_type != "none":
                    if img.shape[2] == 4:
                        msg = "Thresholding cannot be applied after Alpha Channel. Please deselect Alpha Channel."
                        return render_template('module2.html', msg=msg, axis_image_data=axis_image_data, filename=filename)

                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    if thresh_type == 'global_binary':
                        thresh_val = int(request.form.get('thresh_val', 127))
                        _, img = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

                    elif thresh_type == 'global_binary_inv':
                        thresh_val = int(request.form.get('thresh_val', 127))
                        _, img = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

                    elif thresh_type == 'adaptive_mean':
                        block_size = int(request.form.get('block_size', 13))
                        C = int(request.form.get('constant_c', 7))
                        if block_size % 2 == 0:
                            block_size += 1
                        img = cv2.adaptiveThreshold(gray, 255,
                                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                                    cv2.THRESH_BINARY,
                                                    block_size, C)
            except Exception as e:
                msg = f"Error in thresholding operation: {str(e)}"
                return render_template('module2.html', msg=msg, axis_image_data=axis_image_data, filename=filename)

            # === Logical Operations ===
            try:
                if request.form.get('logical') == 'on':
                    if len(img.shape) == 2 or img.shape[2] == 3:
                        mask = np.full(img.shape, 127, dtype=np.uint8)
                        img = cv2.bitwise_and(img, mask)
                    else:
                        msg = "Bitwise operation can only be applied on 1 or 3-channel images."
                        return render_template('module2.html', msg=msg, axis_image_data=axis_image_data, filename=filename)
            except Exception as e:
                msg = f"Error in logical operation: {str(e)}"
                return render_template('module2.html', msg=msg, axis_image_data=axis_image_data, filename=filename)

            # === Alpha Channel ===
            try:
                if request.form.get('alpha') == 'on':
                    b, g, r = cv2.split(img)
                    alpha = np.ones(b.shape, dtype=b.dtype) * 100
                    img = cv2.merge((b, g, r, alpha))
            except Exception as e:
                msg = f"Error in alpha channel operation: {str(e)}"
                return render_template('module2.html', msg=msg, axis_image_data=axis_image_data, filename=filename)

            if 'save' in request.form:
                save_path = os.path.join(upload_folder, "saved_" + filename)
                cv2.imwrite(save_path, img)
                msg = f"Image Saved Successfully as saved_{filename}"

            image_data = convert_to_base64(img)

    return render_template('module2.html',
                           image_data=image_data,
                           axis_image_data=axis_image_data,
                           msg=msg,
                           image_size=image_size,
                           filename=filename)
