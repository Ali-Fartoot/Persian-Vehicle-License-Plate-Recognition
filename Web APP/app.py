from flask import Flask, render_template, request
import os 
from Pipeline import OCR
# webserver gateway interface
app = Flask(__name__ ,template_folder='template')

BASE_PATH = os.getcwd()

UPLOAD_PATH = os.path.join(BASE_PATH, os.path.join('static/Upload'))

# Listener
@app.route('/',methods=['POST','GET'])
def index():
    if request.method == 'POST':
        upload_file = request.files['image_name']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH,filename)
        print(path_save)
        upload_file.save(path_save)
        text = OCR(path_save,filename)

        return render_template('./index.html',upload=True,filename=filename,text=text)

    return render_template('./index.html',upload=False)


if __name__ =="__main__":
    app.run(debug=True)