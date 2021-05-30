from flask import Flask,request,render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobnetv2_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inception_resnet_v2_preprocess
# import matplotlib.pyplot as plt
import numpy as np
import cv2
app = Flask(__name__)

#humans
oct_model = load_model('./models/efficient_net_b5_oct_retinal_scan.h5')
intercranial_model  = load_model('./models/mobnet_v2_RSNA.h5')
pneumonia_model = load_model('./models/pneumonia_vgg_19.h5')
#plants
cassava_model = load_model('./models/model_efficient_net_cassava.h5')
cotton_model = load_model('./models/Inception_resnet_cotton_plant.h5')
#plant_path_model  = load_model('./models/effnet_model_plant_pathology.h6')
tomato_model = load_model('./models/mobnet_v2_Tomato.h5')

@app.route('/')
def home():
    return render_template('Home.html')
@app.route('/dashboard')
def dashboard():
    return render_template('Dashboard.html')
@app.route('/Human')
def Human():
    return render_template('Human.html')
@app.route('/Plants')
def Plants():
    return render_template('Plants.html')
@app.route('/animals')
def animals():
    return render_template('Animals.html')
@app.route('/oct_scan',methods=['GET','POST'])
def oct_scan():
    if request.method == 'POST':
        image = request.files['image'].read()
        npimg = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        image = cv2.resize(image,(128,128))
        image = np.expand_dims(image,axis=0)
        result = list(oct_model.predict(image)[0])
        result = {
            'Choroidal Neovascularization':int(result[0]*100),
            'Diabetic macular edema':int(result[1]*100),
            'Drusen':int(result[2]*100),
            'Normal':int(result[3]*100)
        }
        return render_template('toct.html',data=result)
    return render_template('OCT.html')
@app.route('/intercranial_scan',methods=['GET','POST'])
def intercranial_scan():
    if request.method == 'POST':
        image = request.files['image'].read()
        npimg = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        image = cv2.resize(image,(128,128))
        image = mobnetv2_preprocess(image)
        image = np.expand_dims(image,axis=0)
        result = list(intercranial_model.predict(image)[0])
        result = {
            'Healthy': int(result[0]*100),
            '2 - or More ': int(result[1]*100), 
            '3 - or More': int(result[2]*100), 
            '4 - or More': int(result[3]*100), 
            '5 - or More': int(result[4]*100), 
            '6 - or More': int(result[5]*100)
        }
        return render_template('tintercranial.html',data=result)
    return render_template('IHD.html')

@app.route('/pnuemonia_scan',methods=['GET','POST'])
def pnuemonia_scan():
    if request.method == 'POST':    
        image = request.files['image'].read()
        npimg = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        image = cv2.resize(image,(128,128))
        image = vgg19_preprocess(image)
        image = np.expand_dims(image,axis=0)
        result = list(pneumonia_model.predict(image)[0])
        result = {
            'Normal': 100-int(result[0]*100),
            'Pnuemonia': int(result[0]*100)
        }
        return render_template('tpnuemonia.html',data=result)
    return render_template('Pneumonia.html')
@app.route('/tomato_scan',methods=['GET','POST'])
def tomato_scan():
    if request.method == 'POST':
        image = request.files['image'].read()
        npimg = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        image = cv2.resize(image,(128,128))
        image = mobnetv2_preprocess(image)
        image = np.expand_dims(image,axis=0)
        result = list(tomato_model.predict(image)[0])
        result = {
            'Bacterial spot': int(result[0]*100),
            'Early blight': int(result[1]*100),
            'Late blight': int(result[2]*100),
            'Leaf Mold': int(result[3]*100),
            'Septoria leaf spot': int(result[4]*100),
            'Spider mites Two-spotted spider mite': int(result[5]*100),
            'Target Spot': int(result[6]*100),
            'Yellow Leaf Curl Virus': int(result[7]*100),
            'mosaic virus': int(result[8]*100),
            'healthy': int(result[9]*100)
        }
        return render_template('ttomato.html',data=result)
    return render_template('Tomato.html')
@app.route('/cotton_scan',methods=['GET','POST'])
def cotton_scan():
    if request.method == 'POST':
        image = request.files['image'].read()
        npimg = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        image = cv2.resize(image,(256,256))
        image = inception_resnet_v2_preprocess(image)
        image = np.expand_dims(image,axis=0)
        result = list(cotton_model.predict(image)[0])
        result = {

            'Diseased cotton leaf': int(result[0]*100),
            'Diseased cotton plant': int(result[1]*100),
            'Fresh cotton leaf': int(result[2]*100),
            'Fresh cotton plant': int(result[3]*100)

        }
        return render_template('tcotton.html',data=result)
    return render_template('Cotton.html')
@app.route('/cassava_scan',methods=['GET','POST'])
def cassava_scan():
    if request.method == 'POST':
        image = request.files['image'].read()
        npimg = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        image = cv2.resize(image,(256,256))
        image = np.expand_dims(image,axis=0)
        result = list(cassava_model.predict(image)[0])
        result = {
            'Cassava Bacterial Blight': int(result[0]*100),
            'Cassava Brown Streak Disease ': int(result[1]*100),
            'Cassava Green Mottle': int(result[2]*100),
            'Cassava Mosaic Disease ': int(result[3]*100),
            'Healthy': int(result[4]*100)
        }
        return render_template('tcassava.html',data=result)
    return render_template('Cassava.html')
# @app.route('/plant_scan',methods=['GET','POST'])
# def plant_scan():
#     if request.method == 'POST':
#         image = request.files['image'].read()
#         npimg = np.frombuffer(image, np.uint8)
#         image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
#         image = cv2.resize(image,(128,128))
#         image = np.expand_dims(image,axis=0)
#         result = plant_model.predict(image)
#         return render_template('tplant.html',data=result)
#     return render_template('plant.html')
if __name__ == "__main__":
    app.run(debug=True)