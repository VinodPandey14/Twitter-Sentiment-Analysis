# installing the kaggle library
! pip install kaggle

"""Uploading kaggle.json file for API Key"""

# path of kaggle.json
! mkdir -p ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json

"""Importing Twitter Sentiment Dataset from Kaggle"""

#API to fetch dataset from kaggle
!kaggle datasets download -d kazanova/sentiment140

# extracting the compressed dataset

from zipfile import ZipFile
dataset = '/content/sentiment140.zip'

with ZipFile(dataset, 'r') as zip:
  zip.extractall()
  print('Dataset is extracted')

"""Importing required dependencies"""

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')

print(stopwords.words('english'))

"""Data Processing"""

# loading the data from csc file to pandas dataframe
twitter_data = pd.read_csv('/content/training.1600000.processed.noemoticon.csv', encoding= 'ISO-8859-1')

twitter_data.shape

twitter_data.head()

# naming the columns
column_names = ['target', 'id' , 'date', 'flag', 'user', 'text']
twitter_data = pd.read_csv('/content/training.1600000.processed.noemoticon.csv', names = column_names, encoding= 'ISO-8859-1')
twitter_data.shape

twitter_data.head()

"""Handling Missing Values

"""

twitter_data.isnull().sum()

twitter_data['target'].value_counts()

"""Convert the target '4' to '1' for positive tweet"""

twitter_data.replace({'target': {4: 1}}, inplace=True)
twitter_data['target'].value_counts()

"""0 Label for Negative Tweet

1 Label for Positive Tweet

**Stemming**

Stemming is the process of reduing a word to its root word
"""

port_stem = PorterStemmer()

def stemming(content):
  stemmed_content = re.sub('[^a-zA-Z]', ' ' ,content)
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split()
  stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content = ' '.join(stemmed_content)

  return stemmed_content

twitter_data['stemmed_content'] = twitter_data['text'].apply(stemming) # 50 Mintues to complete this stemming process

twitter_data.head()

print(twitter_data['stemmed_content'])

print(twitter_data['target'])

# Separting the data and label
x = twitter_data['stemmed_content'].values
y = twitter_data['target'].values

print(x)

print(y)

"""Spliting the data to training data and test data"""

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, stratify=y, random_state=2)

print(x.shape, x_train.shape, x_test.shape)

"""Converting the textual data to numerical data"""

vectorizer = TfidfVectorizer()

x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

print(x_train)

print(x_test)

"""**Training the Machine Learning Model (Logistic Regression)**"""

model = LogisticRegression(max_iter=1000)

model.fit(x_train, y_train)

"""**Model Evaluation**

Accuracy Score on Training Data
"""

# accuracy score on the training data

x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)

print('Accuracy Score on the training data : ', training_data_accuracy)

"""Accuracy Score on the Test Data"""

x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test, x_test_prediction)

print( 'Accuracy Score on the test Data : ', test_data_accuracy)

"""Saving the trained model

**Final Model Accuracy = 77.6%**
"""

import pickle

filename = 'trained_model.sav'
pickle.dump(model, open(filename,'wb'))

"""Using the saved model for future prediction"""

#loading the saved model

loaded_model = pickle.load(open('/content/trained_model.sav', 'rb'))

"""Sample Test 1"""

x_new = x_test[200]
print(y_test[200])

prediction = loaded_model.predict(x_new)
print(prediction)

if (prediction[0] == 0):
  print('Negative Tweet')
else:
  print('Positive Tweet')

"""Sample Test 2"""

x_new = x_test[300]
print(y_test[300])

prediction = loaded_model.predict(x_new)
print(prediction)

if (prediction[0] == 0):
  print('Negative Tweet')
else:
  print('Positive Tweet')

"""Sample Test 3"""

x_new = x_test[500]
print(y_test[500])

prediction = loaded_model.predict(x_new)
print(prediction)

if (prediction[0] == 0):
  print('Negative Tweet')
else:
  print('Positive Tweet')
