import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
import cv2
import keras as k

# *************Data augmentation and preprocessing *********************
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    "C:/Users/HP/Documents/Machine Learning/project/skinAnalysis/dataset/Dataset/train",  # Path to training data directory
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical')

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    "C:/Users/HP/Documents/Machine Learning/project/skinAnalysis/dataset/Dataset/test",  # Path to testing data directory
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical')

path = "C:/Users/HP/Documents/Machine Learning/project/skinAnalysis/dataset/Dataset/train"
class_names = sorted(os.listdir(path))
num_classes = len(class_names)

img_size = (192, 192, 3)

print('classes: ', class_names)


labels = []
images = []

print('images:\n')
for cl in class_names:
    print(cl, end=' -> ')
    for img in os.listdir(os.path.join(path, cl)):
        label = np.zeros(num_classes)
        label[class_names.index(cl)] = 1
        labels.append(label)

        image_path = os.path.join(path, cl, img)
        image = np.asarray(cv2.resize(cv2.imread(image_path, cv2.IMREAD_COLOR), img_size[0:2])[:, :, ::-1])
        images.append(image)
    print('done')

    
labels = np.asarray(labels)
images = np.asarray(images)

print(f'\n\nlabels shape: {labels.shape}')
print(f'images shape: {images.shape}')


#*************SPLITTING THE DATA**************

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.15, random_state=42)

print(f'train images shape: {X_train.shape}\ntrain labels shape: {y_train.shape}\n\nvalidation images shape: {X_val.shape}\nvalidation labels shape: {y_val.shape}\n')

#************CNN MODEL ******************
cnn_model = tf.keras.Sequential() 

# Inputs and rescaling
cnn_model.add(tf.keras.layers.Rescaling(scale=1. / 255, input_shape=(img_size)))

# Convolutional block 1
cnn_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
cnn_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

# Convolutional block 2
cnn_model.add(tf.keras.layers.Conv2D(128, (2, 2), activation='relu', padding='same'))
cnn_model.add(tf.keras.layers.Conv2D(128, (2, 2), activation='relu', padding='same'))
cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

# Convolutional block 3
cnn_model.add(tf.keras.layers.Conv2D(256, (2, 2), activation='relu', padding='same'))
cnn_model.add(tf.keras.layers.Conv2D(256, (2, 2), activation='relu', padding='same'))
cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=2))

# Convolutional block 4
cnn_model.add(tf.keras.layers.Conv2D(512, (2, 2), activation='relu', padding='same'))
cnn_model.add(tf.keras.layers.Conv2D(512, (2, 2), activation='relu', padding='same'))
cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
cnn_model.add(tf.keras.layers.Flatten())

# Dense block
cnn_model.add(tf.keras.layers.Dense(256, activation='relu'))
cnn_model.add(tf.keras.layers.Dense(128, activation='relu'))
cnn_model.add(tf.keras.layers.Dense(64, activation='relu'))
cnn_model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

cnn_model.compile(optimizer='Adamax', loss='categorical_crossentropy', metrics=['accuracy'])

cnn_model.summary()


#checkpoint_callback = ModelCheckpoint('cnn_model/model{epoch:02d}')

# early_stopping to prevent overfitting
early_stopping = [k.callbacks.EarlyStopping(monitor='val_loss', patience=10),
                  k.callbacks.ModelCheckpoint(filepath='C:/Users/HP/Documents/Machine Learning/project/skinAnalysis/skin_classifier_model.keras',
                                  monitor='val_accuracy', save_best_only=True)]

# Train the model - fit
history = cnn_model.fit(images, labels, epochs=20, validation_data=(X_val, y_val), callbacks=early_stopping)


#*****************PLOTTING *****************

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))

plt.figure()
plt.plot(epochs, accuracy, label='Training Accuracy')
plt.plot(epochs, loss, label='Training Loss')
plt.legend()
plt.title('Training Accuracy and Loss')

plt.figure()
plt.plot(epochs, val_accuracy, label='Validation Accuracy')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.legend()
plt.title('Validation Accuracy and Loss')

plt.show()
