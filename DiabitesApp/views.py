from django.shortcuts import render , redirect
from .forms import DataForm
import numpy
import pandas as pd 
import  sklearn
import matplotlib.pyplot as plt 
import seaborn
import csv , os
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from django.contrib import messages 

# Create your views here.


def index(request):
    
    return render(request, 'index.html')
    
 
pred = 0 
def predict(request):
    
    

    if request.method == 'POST':
        form = DataForm(request.POST)
        if form.is_valid():
            data = pd.read_csv("static/diabetes.csv")
            x = data.drop('Outcome' , axis=1)
            y = data['Outcome']
            x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.30)
            model = LogisticRegression( solver='lbfgs', max_iter=10000)
            
            model.fit(x_train.values , y_train.values)
            val1 = form.cleaned_data.get("Pregnancies") 
            val2 = form.cleaned_data.get("Glucose")  
            val3 = form.cleaned_data.get("BloodPressure")
            val4 = form.cleaned_data.get("SkinThickness")
            val5 = form.cleaned_data.get("Insulin")
            val6 = form.cleaned_data.get("bmi")
            val7 = form.cleaned_data.get("DiabetesPedigreeFunction")
            val8 = form.cleaned_data.get("Age")
            pred = model.predict([[val1 ,val2 ,val3 ,val4 ,val5 , val6 , val7 ,val8]])
            if pred[0] == 1 :
                messages.success(request, "Positive")
                return redirect('/predict')
            else :
                messages.success(request, "Negative")
                return redirect('/predict')
           
            
    else:
        form = DataForm()
    return render(request, 'predict.html', {'form': form})


    
    