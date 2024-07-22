# train.py

import tensorflow as tf
from model import VGG19Model, compute_content_cost, compute_style_cost
from utils import load_and_process_img, save_image

def style_transfer(content_path, style_path, output_path, img_size=400, content_layer='block5_conv2', style_layers=None, style_layers_weights=None, num_iterations=1000, content_weight=1e3, style_weight=1e-2):
    if style_layers is None:
        style_layers = [
            'block1_conv1',
            'block2_conv1',
            'block3_conv1',
            'block4_conv1',
            'block5_conv1'
        ]
    
    if style_layers_weights is None:
        style_layers_weights = [1.0] * len(style_layers)

    model = VGG19Model(img_size)
    vgg = model.get_model()

    content_img = model.preprocess_image(content_path)
    style_img = model.preprocess_image(style_path)
    generated_img = tf.Variable(content_img, dtype=tf.float32)

    content_layer_output = vgg.get_layer(content_layer).output
    content_model = tf.keras.Model(vgg.input, content_layer_output)
    content_target = content_model(content_img)

    style_model_outputs = [vgg.get_layer(layer).output for layer in style_layers]
    style_model = tf.keras.Model(vgg.input, style_model_outputs)
    style_targets = style_model(style_img)

    optimizer = tf.optimizers.Adam(learning_rate=5.0)

    @tf.function
    def train_step(generated_img):
        with tf.GradientTape() as tape:
            generated_img_processed = tf.keras.applications.vgg19.preprocess_input(generated_img * 255.0)
            generated_img_processed = tf.expand_dims(generated_img_processed, axis=0)
            generated_content_output = content_model(generated_img_processed)
            generated_style_outputs = style_model(generated_img_processed)
            
            J_content = compute_content_cost(content_target, generated_content_output)
            J_style = compute_style_cost(style_targets, generated_style_outputs, style_layers_weights)
            J_total = content_weight * J_content + style_weight * J_style
        
        grad = tape.gradient(J_total, generated_img)
        optimizer.apply_gradients([(grad, generated_img)])
        generated_img.assign(tf.clip_by_value(generated_img, 0.0, 1.0))
        return J_total

    for i in range(num_iterations):
        train_step(generated_img)
        if i % 100 == 0:
            print(f"Iteration {i}/{num_iterations}")

    save_image(generated_img, output_path)

# Example usage
if __name__ == "__main__":
    content_path = "path_to_content_image.jpg"
    style_path = "path_to_style_image.jpg"
    output_path = "path_to_output_image.jpg"
    style_transfer(content_path, style_path, output_path)
