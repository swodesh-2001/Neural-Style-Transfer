import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from utils import *

class NeuralStyleTransfer:
    def __init__(self, content_path, style_path, output_path,iteration = 200, img_size = 50):
        self.content_path = content_path
        self.style_path = style_path
        self.output_path = output_path
        self.img_size = img_size
        self.iteration = iteration
        self.content_layer = [('block5_conv4', 1)]
        self.style_layer = STYLE_LAYERS = [
                            ('block1_conv1', 0.2),
                            ('block2_conv1', 0.2),
                            ('block3_conv1', 0.2),
                            ('block4_conv1', 0.2),
                            ('block5_conv1', 0.2)
                            ]
        self.vgg = load_vgg_model(img_size= self.img_size)

    def generate(self):
        content_image = preprocess_image(self.content_path, self.img_size)
        style_image = preprocess_image(self.style_path, self.img_size)
        generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
        noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
        generated_image = tf.add(generated_image, noise)
        generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)
        generated_image = tf.Variable(generated_image)
        vgg_model_outputs = get_layer_outputs(self.vgg, self.style_layer + self.content_layer)
        preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
        a_C = vgg_model_outputs(preprocessed_content) 
        preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
        a_S = vgg_model_outputs(preprocessed_style)  
        opt = tf.optimizers.Adam(learning_rate=0.02)
        final_image = None
        for i in range(self.iteration):
            with tf.GradientTape() as tape:
                a_G = vgg_model_outputs(generated_image)
                J_style = compute_style_cost(a_S, a_G, STYLE_LAYERS= self.style_layer)
                J_content = compute_content_cost(a_C, a_G)
                J_total = total_cost(J_content, J_style)

            grad = tape.gradient(J_total, generated_image)
            opt.apply_gradients([(grad, generated_image)])
            clipped_image = tf.clip_by_value(generated_image, -1.0, 1.0)


            print(f"Iteration {i} ")
 
                
            
            final_image = clipped_image
        image = tensor_to_image(final_image)
        plt.imshow(image)
        image.save(self.output_path)
        plt.show() 
