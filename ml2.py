import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from flask import Flask,render_template,request
# Load the data
data = pd.read_csv("bankcustomerchurn.csv")
print(data.isnull().sum())#to check if there is any missing valuie in data
print(data.columns)
# Split the data into features and target
x1 = data.drop(['RowNumber', 'CustomerId', 'Surname','Gender','Exited'],axis=1)#dropping these features
y = data["Exited"]
#convert categorical data into 1 and 0
x=pd.get_dummies(x1)
print(x)
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier()
model.fit(X_train.values, y_train.values)
y_pred = model.predict(X_test.values)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy*100)

#accuracy using confusion matrix
mat = metrics.confusion_matrix(y_test, y_pred)
cmp=metrics.ConfusionMatrixDisplay(confusion_matrix=mat,display_labels=[False,True])
cmp.plot()
plt.show()

#deployment using flask
app=Flask(__name__)
@app.route("/")
def home(debug=True):
    return render_template('xyz.html')
@app.route('/predict',methods=["POST"])
def predict(debug=True):
    #Accessing values from form

    if request.method=="POST":
        credit=int(request.form["creditscore"])
        age=int(request.form["age"])
        tenure=int(request.form["tenure"])
        balance=int(request.form["balance"])
        product=int(request.form["no"])
        cc=int(request.form["cc"])
        member=int(request.form["active"])
        salary=float(request.form["salary"])
        geography=request.form["geography"]
        #Checking geography entered by the user

        if(geography=="Germany"):
            Geography_Germany=1
            Geography_France=0
            Geography_Spain=0
        elif(geography=="Spain"):
            Geography_Germany=0
            Geography_France=0
            Geography_Spain=1
        else:
            Geography_Germany=0
            Geography_France=1
            Geography_Spain=0
        
        prediction=model.predict([[credit,age,tenure,balance,product,cc,member,salary,Geography_France,Geography_Germany,Geography_Spain]])
        if prediction==0:
            return render_template('xyz.html',pred="Customer will not exit")
        elif prediction==1:
            return render_template('xyz.html',pred="Customer will exit")

#app will run only when we will be in current file
if __name__=="__main__":
    app.run()
