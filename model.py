# model.py

import tensorflow as tf

class VGG19Model:
    def __init__(self, img_size):
        self.model = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        self.img_size = img_size

    def get_model(self):
        return self.model

    def preprocess_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize(img, (self.img_size, self.img_size))
        img = tf.keras.applications.vgg19.preprocess_input(img)
        return tf.expand_dims(img, axis=0)

    def deprocess_image(self, img):
        img = img.reshape((self.img_size, self.img_size, 3))
        img[:, :, 0] += 103.939
        img[:, :, 1] += 116.779
        img[:, :, 2] += 123.68
        img = img[:, :, ::-1]
        img = tf.clip_by_value(img, 0, 255).numpy().astype('uint8')
        return img

def compute_content_cost(content_output, generated_output):
    return tf.reduce_mean(tf.square(content_output - generated_output))

def gram_matrix(A):
    channels = int(A.shape[-1])
    a = tf.reshape(A, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

def compute_layer_style_cost(style_output, generated_output):
    S = gram_matrix(style_output)
    G = gram_matrix(generated_output)
    return tf.reduce_mean(tf.square(S - G))

def compute_style_cost(style_outputs, generated_outputs, style_layers_weights):
    J_style = 0
    for style_output, generated_output, weight in zip(style_outputs, generated_outputs, style_layers_weights):
        J_style += weight * compute_layer_style_cost(style_output, generated_output)
    return J_style
