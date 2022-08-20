#Import libraries
from email import message
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import string

#Loading and wrangling the dataset
df = pd.read_csv('spam.csv')
df.rename(columns={'v1': 'spam', 'v2': 'text'}, inplace=True)
df = df[['text', 'spam']]
df['spam'] = df['spam'].replace('ham', 0)
df['spam'] = df['spam'].replace('spam', 1)
df.drop_duplicates(inplace=True)

#Cleaning text
def clean_text(text):
    #Removing the punctuation
    punct = [word for word in text if text not in string.punctuation]
    punct = ''.join(punct)

    #Removing the stopwords
    cleaned = [word for word in punct.split() if word.lower() not in stopwords.words('english')]
    return cleaned

#Convert text to a token matrix
from sklearn.feature_extraction.text import CountVectorizer
messages = CountVectorizer(analyzer=clean_text).fit_transform(df['text'])

#Obtaining the models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#Performing a 10-fold cross validation to obtain the accuracy of each classifier
from sklearn.model_selection import cross_val_score
naive_bayes_accuracy = np.mean(cross_val_score(MultinomialNB(), messages, df['spam'], scoring = "accuracy", cv = 10))
print("The 10-fold cross validation accuracy for the Naive Bayes Classifier is " + 
    str(100*round(naive_bayes_accuracy, 3)) + "%.")

LR_accuracy = np.mean(cross_val_score(LogisticRegression(), messages, df['spam'], scoring = "accuracy", cv = 10))
print("The 10-fold cross validation accuracy for Logistic Regression is " + 
    str(100*round(LR_accuracy, 3)) + "%.")

SVM_linear_accuracy = np.mean(cross_val_score(SVC(kernel = "linear"), messages, df['spam'], scoring = "accuracy", cv = 10))
print("The 10-fold cross validation accuracy for linear SVM is " + 
    str(100*round(SVM_linear_accuracy, 3)) + "%.")

SVM_radial_accuracy = np.mean(cross_val_score(SVC(kernel = "rbf"), messages, df['spam'], scoring = "accuracy", cv = 10))
print("The 10-fold cross validation accuracy for radial SVM is " + 
    str(100*round(SVM_radial_accuracy, 3)) + "%.")

SVM_polynomial_accuracy = np.mean(cross_val_score(SVC(kernel = "poly"), messages, df['spam'], scoring = "accuracy", cv = 10))
print("The 10-fold cross validation accuracy for polynomial SVM is " + 
    str(100*round(SVM_polynomial_accuracy, 3)) + "%.")

SVM_sigmoid_accuracy = np.mean(cross_val_score(SVC(kernel = "sigmoid"), messages, df['spam'], scoring = "accuracy", cv = 10))
print("The 10-fold cross validation accuracy for sigmoid SVM is " + 
    str(100*round(SVM_sigmoid_accuracy, 3)) + "%.")

print()
print("------------------------------------------------------------")
print()

#Conclusion
print("While most of the classifiers do very well, predicting the correct category 95-97 " + 
    "percent of the time, the Logistic regression, linear SVM, and sigmoid SVM, do the best job " + 
    "with ~ 97 percent accuract each.")