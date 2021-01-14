"""Name: Aditya Vashista 
Batch: Coe 2
Roll Number: 101703039
Problem: Plant Seedlings Classification"""

#importing libraries
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing import image
from tensorflow.keras.layers import Dense, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import  Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

#generating training and testing data of images available
datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.3,
                                   zoom_range = 0.3,rotation_range=0.3,
                                   width_shift_range=0.3,height_shift_range=0.3,
                                   horizontal_flip = True,vertical_flip=True,
                                   validation_split=0.3)

train_set=datagen.flow_from_directory("C:\\Users\\aditya\\Downloads\\train",
                                      target_size=(256,256),batch_size=32,class_mode='categorical',
                                      subset='training')
test_set=datagen.flow_from_directory("C:\\Users\\aditya\\Downloads\\train",
                                      target_size=(256,256),batch_size=32,class_mode='categorical',
                                      subset='validation')

#Creating Sequential CNN + ANN model
#For each CNN layer batchnormalization,drop out and max pooling is also appliead
model=Sequential()
#CNN LAYER 1: INPUT LAYER
model.add(Conv2D(64,(3,3),padding='same',input_shape=(256,256,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2,strides=2,padding='valid'))
model.add(Dropout(0.25))
#CNN LAYER 2
model.add(Conv2D(128,(5,5),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#CNN LAYER 3
model.add(Conv2D(256,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2,strides=2,padding='valid'))
model.add(Dropout(0.25))
#CNN LAYER 4
model.add(Conv2D(512,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2,strides=2,padding='valid'))
model.add(Dropout(0.25))
#CNN LAYER 5
model.add(Conv2D(512,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2,strides=2,padding='valid'))
model.add(Dropout(0.25))
#FLATTENING
model.add(Flatten())
#ANN LAYER 1
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
#OUTPUT LAYER(ANN-2) with 12 outputs
model.add(Dense(12,activation='softmax'))
#creating optimizer
opt=Adam(lr=0.0005)
#model compilation
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['categorical_accuracy','accuracy'])
#model summary
model.summary()


#Total epochs used for training=38
#Epochs were devided as: 38=16+8+10+4 with different validation splits:(2,3)

#MODEL TRAINING
#epochs=4 //last epocs set
epochs=38
steps_per_epoch=train_set.n//train_set.batch_size
test_steps=test_set.n//test_set.batch_size

model.fit(x=train_set,epochs=epochs,validation_data=test_set,validation_steps=test_steps)

# saving model
model.save("model.h5")
del model
#loading model
model=load_model('model.h5')

#testing and verification of various images manually to check accuracy
testImage1=image.load_img('train//Small-flowered Cranesbill//0b26e2d09.png',target_size=(256,256))
testImage1=image.img_to_array(testImage1)
testImage1=np.expand_dims(testImage1,0)
print(model.predict(testImage1))
print(np.argmax(model.predict(testImage1)))


#TESTING
#DEFINING 12 classes as per requirements
classes=['Black-grass','Charlock','Cleavers','Common Chickweed','Common wheat','Fat Hen',
         'Loose Silky-bent','Maize','Scentless Mayweed','Shepherds Purse',
         'Small-flowered Cranesbill','Sugar beet']

#reading sample file for testing 
sam=pd.read_csv('sample_submission.csv')
#dataframe to be stored in output file
df=pd.DataFrame(columns=['file','species'])

#computing output
for i in range(len(sam)):
    testImage=image.load_img('test//'+str(sam.iloc[i][0]),target_size=(256,256)) #loading a single image
    #necessary steps to convert images into suitable input format for model
    testImage=image.img_to_array(testImage)
    testImage=np.expand_dims(testImage,0)
    #getting index of max output from list of 12 outputs from the model after prediction
    pred=np.argmax(model.predict(testImage))
    #assigning corresponding class
    pred=classes[pred]
    #adding the output with image name in dataframe
    df=df.append({'file':sam.iloc[i][0],'species':pred},ignore_index=True)
 
#saving output to new file    
df.to_csv('submission.csv',index=False)

