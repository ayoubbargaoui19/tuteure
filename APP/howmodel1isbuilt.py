
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Churn_Modelling.csv')
test = pd.read_csv('testtest1.csv')
data_re=dataset[dataset['Exited']==1]
data_re.set_index('RowNumber',inplace=True)
print(data_re)
data_re.to_csv('data_re.csv')
X = dataset.iloc[:, 3:13].values
X_test=test.iloc[:, 3:13].values

y= dataset.iloc[:, 13].values
y_test= test.iloc[:, 13].values

print(X)
print(y)
print(X_test)
print(y_test)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

labelencoder_X_3 = LabelEncoder()#creating label encoder object no. 1 to encode region name(index 1 in features)
X_test[:, 1] = labelencoder_X_3.fit_transform(X_test[:, 1])#encoding region from string to just 3 no.s 0,1,2 respectively
labelencoder_X_4 = LabelEncoder()
X_test[:, 2] = labelencoder_X_4.fit_transform(X_test[:, 2])#encoding Gender from string to just 2 no.s 0,1(male,female) respectively

onehotencoder2 = OneHotEncoder(categorical_features = [1])
X_test= onehotencoder2.fit_transform(X_test).toarray()
X_test = X_test[:, 1:]


from sklearn.model_selection import train_test_split
X_train=X
y_train=y


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import load_model

from keras.models import Sequential#For building the Neural Network layer by layer
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV




#------2)Defining a Graph
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
classifier=KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size':[2,16],
              'nb_epoch':[500],
              'optimizer':['adam']}
grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,
                         scoring='accuracy',
                         cv=10 )

history=grid_search.fit(X_train,y_train)
best_parameters=grid_search.best_params_
best_accuracy=grid_search.best_score_


classifier.save('my_model.h5')  # creates a HDF5 file 'my_model.h5


#Lets plot the increase of accuracy as we increase the number of training epochs
import matplotlib.pyplot as plt
print(history.history.keys())
# summarize history for accuracy
plt.figure(figsize=(12,12))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#classifier=load_model('my_model.h5') 
y_pred = history.predict(X_test)
#y_pred = (y_pred > 0.5)
#y_pred=[1 if i>0.5 else 0 for i in y_pred ]
print(y_pred)
dff=pd.read_csv("testtest1.csv")
dff['Exited']=y_pred
dff.set_index('RowNumber',inplace=True)
dff.sort_values('Exited',ascending=False,inplace=True)
dff.to_csv('testtest1.csv') #output file

