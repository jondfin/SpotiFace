from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage
import os

cApp = ClarifaiApp(api_key='3b338a62ceb14a79a33e8aeb668de687')
app = Flask(__name__, template_folder='template')
root = os.path.dirname(os.path.abspath(__file__))

def predict(filename):
    #Predict the emotions from the image and return a song
    model = cApp.models.get('faces')
    model.train()
    image = ClImage(file_obj=open(os.curdir+'/images/'+filename, 'rb'))
    #Prediction value
    model.model_version='f2dd3977d1124cac862f187b797585f0'
    response = model.predict([image])
    print(response)

    #Happy
    print('Happy value ' + str(response['outputs'][0]['data']['concepts'][0]['value']))
    happyVal = response['outputs'][0]['data']['concepts'][0]['value']

    #Sad
    print('Sad value ' + str(response['outputs'][0]['data']['concepts'][1]['value']))
    sadVal = response['outputs'][0]['data']['concepts'][1]['value']

    #Mad
    print('Mad value ' + str(response['outputs'][0]['data']['concepts'][2]['value']))
    madVal = response['outputs'][0]['data']['concepts'][2]['value']

    #Calculate values and return appropriate html element
    if(happyVal > sadVal and happyVal > madVal):
        return render_template('Face2Song.html', Happy=True)
    if(madVal > happyVal and madVal > sadVal):
        return render_template('Face2Song.html', Mad=True)
    if(sadVal > happyVal and sadVal > madVal):
        return render_template('Face2Song.html', Sad=True)


@app.route("/")
def home():
    return render_template('home.html')

@app.route("/Upload/", methods=['POST', 'GET'])
def upload():
    #If the folder does not exist, create it
    target = os.path.join(root, 'images/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)

    #Save file
    file = request.files['file']
    filename = secure_filename(file.filename)
    print(filename)
    file.save(target + filename)
    return predict(filename)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
