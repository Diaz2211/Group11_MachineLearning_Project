import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

# dataset
dataset_path = "source_code/Data"

if not os.path.exists(dataset_path):
    print("Không tìm thấy dataset:", dataset_path)
    exit()

# tien xu ly du lieu
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    brightness_range=[0.4, 1.6],
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(30, 30),
    batch_size=32,
    class_mode='categorical',
    subset='training' 
)

valid_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
valid_generator = valid_datagen.flow_from_directory(
    dataset_path,
    target_size=(30, 30),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# kien truc model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(30,30,3)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(38, activation='softmax') 
])

# train
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model_v2.h5', monitor='val_accuracy', save_best_only=True, mode='max')

history = model.fit(train_generator, epochs=30, validation_data=valid_generator, callbacks=[checkpoint])
