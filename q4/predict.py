#import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
#from PIL import Image
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from pathlib import Path
import pathlib


def main():
    colors_code = {
    'pink': 1,
    'purple': 2,
    'yellow': 3,
    'orange': 4,
    'white': 5,
    'silver': 6,
    'grey': 7,
    'black': 8,
    'red': 9,
    'brown': 10,
    'green': 11,
    'blue': 12
    }
    batch_size = 32
    img_height = 180
    img_width = 180
    data_dir = pathlib.Path('train')
    #image_count = len(list(data_dir.glob('*/*.jpg')))
    #print(image_count)
    #blacks = list(data_dir.glob('black/*'))
    #Image.open(open("path/to/file", 'rb'))
    #PIL.Image.open(open(str(blacks[0]), 'rb'))
      
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)



    class_names = train_ds.class_names
    print(class_names)

    num_classes = 12

    model = Sequential([
      layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
    ])
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    model.summary()

    epochs=10
    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs
    )
    
    model.save("my_model")
    
    model = keras.models.load_model("my_model")
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
              
    color_ids = []
    file_names = []
    for filename in os.listdir('test'):
        pic_path = 'test/' + filename
        img = keras.preprocessing.image.load_img(
            pic_path, target_size=(img_height, img_width)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        file_names += [filename]
        color_id = colors_code[class_names[np.argmax(score)]]
        color_ids += [color_id]
    write_ans_to_csv(file_names, color_ids)



main()
def write_ans_to_csv(file_names, color_ids):
    with open('ans.csv', 'w', newline='') as csvfile:
        fieldnames = ['file_name', 'color_id']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for index, c in enumerate(color_ids):
            writer.writerow({'file_name': file_names[index], 'color_id': c})