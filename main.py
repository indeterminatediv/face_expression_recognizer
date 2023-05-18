from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
import os



train_data_dir = '/home/ravi/PROJECT_OM/dataset/train/'
validation_data_dir = '/home/ravi/PROJECT_OM/dataset/test/'

# data loader and preprocessing

train_datagen = ImageDataGenerator(
                      rescale=1./255,
                      rotation_range=30,
                      shear_range=0.3,
                      zoom_range=0.3,
                      horizontal_flip=True,
                      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                         train_data_dir,
                         color_mode='grayscale',
                         target_size=(48,48),
                         batch_size=32,
                         class_mode='categorical',
                         shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
                         validation_data_dir,
                         color_mode='grayscale',
                         target_size=(48,48),
                         batch_size=32,
                         class_mode='categorical',
                         shuffle=True)

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise' ]
img, label = train_generator.__next__()

#model

model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), acivation='relu', input_shape=(48,48,1)))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(7, activation='softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
print(model.summary())





