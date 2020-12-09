from flask import Flask
from flask import request
from flask import render_template


import os
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.backend import clear_session


app = Flask(__name__)

upload_folder = 'static/image'

def predict(image_path):
    model = None
    prediction = None
    model = load_model('model.h5')
    img = image.load_img(image_path, target_size=(224,224))
    #processing the image
    x = image.img_to_array(img)
    x = x/255  #scaling
    x = np.expand_dims(x, axis=0)
    
    x = preprocess_input(x)
    prediction = model.predict(x)
    clear_session()
    prediction = np.argmax(prediction, axis=1)
    return prediction

@app.route('/', methods=['GET','POST'])
def hello():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_location = os.path.join(
                upload_folder,
                image_file.filename
            )
            image_file.save(image_location)
            pred = predict(image_location)[0]
            return render_template('index.html', prediction=pred, image_loc=image_location)
    return render_template('index.html', prediction=0, image_loc = None)




if __name__ == '__main__':
    app.run(debug=True)