import tensorflow as tf
from model import vgg_layers, get_content_loss, get_style_loss, gram_matrix
from utils import load_and_process_img, deprocess_img, imshow

class NeuralStyleTransfer:
    def __init__(self, content_path, style_path):
        self.content_path = content_path
        self.style_path = style_path
        self.content_layers = ['block5_conv2'] 
        self.style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        self.num_content_layers = len(self.content_layers)
        self.num_style_layers = len(self.style_layers)
        self.model = self.build_model()

    def build_model(self):
        style_outputs = [layer for layer in self.style_layers]
        content_outputs = [layer for layer in self.content_layers]
        model_outputs = style_outputs + content_outputs
        return vgg_layers(model_outputs)

    def compute_loss(self, outputs):
        style_outputs = outputs[:self.num_style_layers]
        content_outputs = outputs[self.num_style_layers:]

        style_loss = 0
        content_loss = 0

        # Adjusting the sizes for style layers to match target sizes
        for style_output, target in zip(style_outputs, self.style_targets):
            style_loss += get_style_loss(style_output, target)

        style_loss *= 1.0 / self.num_style_layers

        for content_output, target in zip(content_outputs, self.content_targets):
            content_loss += get_content_loss(content_output, target)

        content_loss *= 1.0 / self.num_content_layers

        total_loss = style_loss + content_loss
        return total_loss

    def compute_grads(self, cfg):
        with tf.GradientTape() as tape:
            all_loss = self.compute_loss(self.model(cfg['init_image']))
        total_loss = all_loss
        return tape.gradient(total_loss, cfg['init_image']), total_loss

    def run_style_transfer(self, num_iterations=1000, content_weight=1e3, style_weight=1e-2):
        self.content_image = load_and_process_img(self.content_path)
        self.style_image = load_and_process_img(self.style_path)

        self.model = self.build_model()

        style_features = self.model(self.style_image)
        content_features = self.model(self.content_image)

        self.style_targets = [gram_matrix(style_feature) for style_feature in style_features[:self.num_style_layers]]
        self.content_targets = content_features[self.num_style_layers:]

        init_image = tf.Variable(self.content_image, dtype=tf.float32)
        opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
        
        cfg = {
            'model': self.model,
            'init_image': init_image,
            'style_targets': self.style_targets,
            'content_targets': self.content_targets,
            'content_weight': content_weight,
            'style_weight': style_weight
        }

        best_loss, best_img = float('inf'), None

        for i in range(num_iterations):
            grads, total_loss = self.compute_grads(cfg)
            opt.apply_gradients([(grads, init_image)])
            clipped = tf.clip_by_value(init_image, -1.0, 1.0)
            init_image.assign(clipped)

            if total_loss < best_loss:
                best_loss = total_loss
                best_img = deprocess_img(init_image.numpy())

            if i % 100 == 0:
                print(f"Iteration: {i}, Loss: {total_loss}")

        return best_img, best_loss
