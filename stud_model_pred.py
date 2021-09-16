from flask import Flask, render_template, request
import numpy as np
app = Flask(__name__)
from tensorflow.keras.models import load_model
model=load_model("studs_pred_updated.h5")

@app.route('/')
def home():
    return render_template("home.html")
@app.route('/upload')
def upload():
    return render_template("upload.html")
@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = np.array(int_features).reshape(1,10,1)
    prediction = model.predict_classes(final_features)
    L_collection={0:"ARTS",1:"DESIGN",2:"ENGINEERING",3:"MALAYALAM",4:"MBBS",5:"SPORTS",6:"TRANSLATION"}
    result=L_collection[prediction[0]]
    return render_template("result.html",prediction_text=f"THE BEST CHOICE IS :  {result} ")
if __name__=="__main__":
    app.run(debug=True)

