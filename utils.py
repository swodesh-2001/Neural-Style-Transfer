# utils.py

import tensorflow as tf
from PIL import Image
import numpy as np

def load_and_process_img(path_to_img, img_size):
    img = Image.open(path_to_img)
    img = img.resize((img_size, img_size))
    img = np.array(img)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

def save_image(img, output_path):
    img = img.numpy()
    img = img[0]
    img = tf.keras.applications.vgg19.deprocess_input(img)
    img = tf.clip_by_value(img, 0, 255).astype('uint8')
    Image.fromarray(img).save(output_path)
