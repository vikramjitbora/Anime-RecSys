from flask import Flask,render_template,request
from pipeline.prediction_pipeline import hybrid_recommendation
import sys
from src.custom_exception import CustomException

app = Flask(__name__)

@app.route('/' , methods=['GET','POST'])
def home():
    recommendations = None

    if request.method == 'POST':
        try:
            user_id = int(request.form["userID"])

            recommendations = hybrid_recommendation(user_id)
        except Exception as e:
            raise CustomException(str(e), sys)


    return render_template('index.html' , recommendations=recommendations)

if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0',port=8000)