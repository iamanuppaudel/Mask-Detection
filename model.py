#%%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Activation, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time
import numpy as np 



# %%
IMG_SIZE= 80
train_data_path= 'Dataset/Train'
test_data_path= 'Dataset/Test'
num_classes=2

#generating datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_data =ImageDataGenerator(horizontal_flip=True,
                               rescale=1./255,
                               shear_range=0.2,
                               zoom_range=0.2
                               )

training_set = train_data.flow_from_directory(train_data_path,
                                              target_size=(IMG_SIZE,IMG_SIZE),
                                              batch_size=50,
                                              classes=['WithMask','WithoutMask']
                                              )

test_data= ImageDataGenerator(rescale=1./255
                             )

testing_set = test_data.flow_from_directory(test_data_path,
                                            target_size=(IMG_SIZE,IMG_SIZE),
                                            batch_size=50,
                                            classes=['WithMask','WithoutMask']                                          
                                           )
#defining the model
def model():
    
    model = Sequential()

    #layers
    # input layer
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    #first hidden layer
    model.add(Conv2D(80, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    #second hidden layer
    model.add(Conv2D(100, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    #third hidden layer
    model.add(Conv2D(120, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    

    model.add(Flatten())

    #Dense layer
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))

    #final layer
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    return model

NAME= "Mask_Detection_CNN_{}".format(int(time.time()))
tensorboard= TensorBoard(log_dir="logs\{}".format(NAME))

model=model()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit_generator(
    training_set,
    steps_per_epoch=200,
    epochs=15,
    validation_data=testing_set,
    validation_steps=20,
    callbacks=[tensorboard]
)

model.save(NAME)















# %%
