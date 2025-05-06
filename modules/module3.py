import cv2
import os
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import render_template

def convert_to_base64(image):
    _, buf = cv2.imencode('.jpg', image)
    return base64.b64encode(buf).decode('utf-8')

def save_axis_image(img, upload_folder):
    h, w = img.shape[:2]
    x_step = max(w//10, 50)
    y_step = max(h//10, 50)
    plt.figure(figsize=(8,8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.xticks(np.arange(0, w, step=x_step))
    plt.yticks(np.arange(0, h, step=y_step))
    plt.grid(False)
    path = os.path.join(upload_folder, "module3_axis_image.jpg")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return path

def plot_combined_histogram(img, upload_folder):
    plt.figure()
    for idx, col in enumerate(('b','g','r')):
        hist = cv2.calcHist([img],[idx],None,[256],[0,256])
        plt.plot(hist, color=col)
    plt.title("Combined RGB Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    hist_path = os.path.join(upload_folder, "combined_histogram.jpg")
    plt.savefig(hist_path, bbox_inches='tight')
    plt.close()
    with open(hist_path,'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def process(request, upload_folder):
    image_data = axis_image_data = histogram_data = segmented_data = None
    msg = image_size = filename = None

    if request.method=='POST':
        # 1) Handle upload or reuse
        if 'image' in request.files:
            f = request.files['image']
            if f:
                filename = f.filename
                fp = os.path.join(upload_folder, filename)
                f.save(fp)
        else:
            filename = request.form.get('filename')

        if filename:
            fp = os.path.join(upload_folder, filename)
            img = cv2.imread(fp)

            # 2) Axis preview
            axis_path = save_axis_image(img, upload_folder)
            with open(axis_path,'rb') as f:
                axis_image_data = base64.b64encode(f.read()).decode('utf-8')

            # 3) Show size
            if request.form.get('show_size')=='on':
                h,w = img.shape[:2]
                image_size = f"Width: {w}px, Height: {h}px"

            # 4) Histogram?
            if request.form.get('histogram')=='on':
                histogram_data = plot_combined_histogram(img, upload_folder)

            # 5) Color Segmentation?
            if request.form.get('segment')=='on':
                space = request.form.get('space','HSV')
                choice = request.form.get('selected_color','custom')

                if space=='HSV':
                    converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    # Predefined HSV bounds
                    bounds = {
                      'red':    ((0,120,70),(10,255,255)),
                      'green':  ((36,100,100),(86,255,255)),
                      'blue':   ((94,80,2),(126,255,255)),
                      'yellow': ((20,100,100),(30,255,255)),
                      'orange': ((10,100,20),(25,255,255)),
                      'purple': ((130,50,50),(160,255,255)),
                      'white':  ((0,0,200),(179,30,255)),
                      'black':  ((0,0,0),(179,255,50)),
                      'gray':   ((0,0,50),(179,30,200)),
                    }
                    if choice in bounds:
                        lower, upper = bounds[choice]
                    else:
                        lower = (
                          int(request.form.get('l_h',0)),
                          int(request.form.get('l_s',0)),
                          int(request.form.get('l_v',0))
                        )
                        upper = (
                          int(request.form.get('u_h',179)),
                          int(request.form.get('u_s',255)),
                          int(request.form.get('u_v',255))
                        )
                else:
                    converted = img
                    # BGR manual only
                    lower = (
                      int(request.form.get('l_b',0)),
                      int(request.form.get('l_g',0)),
                      int(request.form.get('l_r',0))
                    )
                    upper = (
                      int(request.form.get('u_b',255)),
                      int(request.form.get('u_g',255)),
                      int(request.form.get('u_r',255))
                    )

                mask = cv2.inRange(converted,
                                  np.array(lower),
                                  np.array(upper))
                result = cv2.bitwise_and(img, img, mask=mask)

                # Encode mask and result
                _, mbuf = cv2.imencode('.jpg', mask)
                segmented_data = convert_to_base64(result)

            # 6) Save?
            if 'save' in request.form:
                outp = os.path.join(upload_folder,"saved_"+filename)
                cv2.imwrite(outp,img)
                msg = f"Saved as saved_{filename}"

            # Always show final image
            image_data = convert_to_base64(img)

    return render_template('module3.html',
                           image_data=image_data,
                           axis_image_data=axis_image_data,
                           histogram_data=histogram_data,
                           segmented_data=segmented_data,
                           msg=msg,
                           image_size=image_size,
                           filename=filename)
