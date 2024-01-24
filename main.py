import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

dataset = pd.read_csv("D:/MACHINE LEARNING PROJECTS/Breast-Cancer-Detection-Using-Logistic-Regression/breast_cancer.csv")
X= dataset.iloc[:,1:-1]
y = dataset.iloc[:,-1]

'''print(dataset.head())
print(dataset.isnull().sum())'''

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
classifier = LinearRegression()
classifier.fit(X_train,y_train)

a=int(input("Clump Thickness: "))
b=int(input("Uniformity of Cell Size: "))
c=int(input("Uniformity of Cell Shape: "))
d=int(input("Marginal Adhesion: "))
e=int(input("Single Epithelial Cell Size: "))
f=int(input("Bare Nuclei: "))
g=int(input("Bland Chromatin: "))
h=int(input("Normal Nucleoli: "))
i=int(input("Mitoses: "))

y_pred = classifier.predict([[a,b,c,d,e,f,g,h,i]])
print(y_pred)
print("\n\n\n")

if(y_pred > 0 and y_pred <= 2.5):
    print("You Are Suffering From Benign\n\nIn the context of tumors, benign growths are typically well-defined, slow-growing, and do not invade surrounding tissues.\nThey also do not metastasize, meaning they do not spread to other parts of the body through the bloodstream or lymphatic system.\nWhile benign tumors may still cause problems depending on their size and location, they are not considered cancerous.")
elif(y_pred > 2.5 and y_pred <= 3.5):
    print("You Must Consult Doctor")
else:
    print("You Are Suffering From Malignant.\n\nThe term malignant is commonly used in medicine to describe cells or tumors that are harmful, aggressive, and have the potential to invade nearby tissues and spread to other parts of the body.\nMalignant cells are typically associated with cancer.\nIn the context of cancer, a malignant tumor is made up of cells that can grow uncontrollably, invade surrounding tissues, and metastasize (spread) to other parts of the body through the bloodstream or lymphatic system.\nBenign conditions or lesions can also refer to non-cancerous abnormalities in organs or tissues that do not pose a significant threat to health.")


print("\n\n\n")
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy of the model is : {:.2f} %".format(accuracies.mean()*100))

