import tensorflow as tf
import matplotlib.pyplot as plt

data_dir = "C:/Users/HP/Documents/Machine Learning/project/skinAnalysis/dataset/Skin_dataset"
# Image parameters
img_size = (128, 128)
batch_size = 16

# Load and preprocess dataset
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
rescale=1./255,
validation_split=0.2
)
train_data = datagen.flow_from_directory(
data_dir,
target_size=img_size,
batch_size=batch_size,
class_mode='categorical',
subset='training'
)
val_data = datagen.flow_from_directory(
data_dir,
target_size=img_size,
batch_size=batch_size,
class_mode='categorical',
subset='validation'
)
# Define CNN model
model = tf.keras.Sequential([
tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dense(train_data.num_classes, activation='softmax')
])
model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
# Train model
history = model.fit(train_data, epochs=10, validation_data=val_data)

# Save model
model.save("skin_classifier_model.keras")

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy Over Epochs")
plt.show()
