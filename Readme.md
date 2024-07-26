# Neural Style Transfer

## Table of Contents
1. [Introduction](#introduction)
    - [What is Neural Style Transfer?](#what-is-neural-style-transfer)
    - [Applications of Neural Style Transfer](#applications-of-neural-style-transfer)
2. [Theory](#theory)
    - [Content and Style Representations](#content-and-style-representations)
    - [Loss Functions](#loss-functions)
        - [Content Loss](#content-loss)
        - [Style Loss](#style-loss)
        - [Total Loss](#total-loss)
3. [Project Structure](#project-structure)
    - [`utils.py`](#utilspy)
    - [`model.py`](#modelpy)
    - [`nst.py`](#nstpy)
    - [`main.py`](#mainpy)
4. [Detailed Code Explanation](#detailed-code-explanation)
    - [Loading and Processing Images](#loading-and-processing-images)
    - [Building the VGG Model](#building-the-vgg-model)
    - [Computing the Losses](#computing-the-losses)
    - [Running the Style Transfer](#running-the-style-transfer)
5. [Running the Script](#running-the-script)
6. [References](#references)

## 1. Introduction

### What is Neural Style Transfer?
Neural Style Transfer (NST) is a technique that takes two images—a content image and a style image—and blends them together so that the output image looks like the content image but "painted" in the style of the style image. It uses convolutional neural networks (CNNs) to achieve this effect.

### Applications of Neural Style Transfer
NST has been used in various applications including:
- Artistic image generation
- Photo and video editing
- Creating visually appealing content for social media and marketing

## 2. Theory

### Content and Style Representations
To understand NST, we need to grasp how CNNs can be used to extract features from images. The VGG network, a deep CNN, is often used for this purpose. The intermediate layers of this network can be used to capture both content and style representations of an image.

### Loss Functions
The core idea of NST is to define and minimize a loss function that blends the content of one image with the style of another.

#### Content Loss
The content loss measures how different the content of the generated image is from the content image. It is typically calculated as the mean squared error (MSE) between the feature representations of the content image and the generated image at a certain layer.

\[
\mathcal{L}_{\text{content}}(C, G) = \sum_{i,j} \left( F_{ij}^{C} - F_{ij}^{G} \right)^2
\]

where $F_{ij}^{C}$ and $F_{ij}^{G}$ are the feature representations of the content image and generated image, respectively.

#### Style Loss
The style loss measures how different the style of the generated image is from the style image. It is calculated using the Gram matrix of the feature representations. The Gram matrix captures the correlations between the different feature maps.

\[
\mathcal{L}_{\text{style}}(S, G) = \sum_{l} w_{l} \sum_{i,j} \left( G_{ij}^{S} - G_{ij}^{G} \right)^2
\]

where $G_{ij}^{S}$ and $G_{ij}^{G}$ are the Gram matrices of the style image and generated image, and $w_{l}$ are the weights for each layer.

#### Total Loss
The total loss is a weighted sum of the content loss and the style loss.

\[
\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{content}} + \beta \mathcal{L}_{\text{style}}
\]

where $\alpha$ and $\beta$ are the weights for the content and style loss, respectively.

## 3. Project Structure
The project is divided into four main modules:

- `utils.py`: Helper functions for image loading and processing.
- `model.py`: Defines the VGG model and its layers for style and content extraction.
- `nst.py`: Implements the Neural Style Transfer logic.
- `main.py`: The entry point for the script, handling argument parsing and running the transfer.

## 4. Detailed Code Explanation

### Loading and Processing Images (`utils.py`)
This module contains functions to load and preprocess images for the VGG model.

#### Loading Images
```python
def load_img(path_to_img):
    max_dim = 512
    img = Image.open(path_to_img)
    long = max(img.size)
    scale = max_dim / long
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.ANTIALIAS)
    img = np.array(img)

    # We need to broadcast the image array such that it has a batch dimension
    img = np.expand_dims(img, axis=0)
    return img
```

This function loads an image from a specified path and resizes it so that its largest dimension is 512 pixels. The image is then expanded to have a batch dimension, making it compatible with the VGG model input.

#### Preprocessing and Deprocessing
```python
def load_and_process_img(path_to_img):
    img = load_img(path_to_img)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3
    if len(x.shape) != 3:
        raise ValueError("Input to deprocess image must be an image of dimension [1, height, width, channel] or [height, width, channel]")

    # perform the inverse of the preprocessing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x
```

The `load_and_process_img` function preprocesses the image by converting the image from RGB to BGR and subtracting the mean RGB value for each pixel. The `deprocess_img` function reverses this process.

### Building the VGG Model (`model.py`)
This module defines the VGG model and its layers for style and content extraction.

```python
def vgg_layers(layer_names):
    """ Creates a VGG model that returns a list of intermediate output values."""
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model
```

This function creates a VGG model that outputs the intermediate layers specified in `layer_names`.

### Computing the Losses (`model.py`)
The content and style loss functions are defined here.

```python
def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / num_locations

def get_style_loss(base_style, gram_target):
    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)

    return tf.reduce_mean(tf.square(gram_style - gram_target))
```

The `get_content_loss` function computes the MSE between the content representations. The `gram_matrix` function computes the Gram matrix, and the `get_style_loss` function computes the MSE between the Gram matrices of the style representations.

### Running the Style Transfer (`nst.py`)
This module contains the main logic for running the style transfer.

#### Initialization
```python
class NeuralStyleTransfer:
    def __init__(self, content_path, style_path):
        self.content_path = content_path
        self.style_path = style_path
        self.content_layers = ['block5_conv2'] 
        self.style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        self.num_content_layers = len(self.content_layers)
        self.num_style_layers = len(self.style_layers)
        self.model = self.build_model()
```

This class initializes the content and style images, defines the layers for content and style extraction, and builds the VGG model.

#### Building the Model
```python
def build_model(self):
    style_outputs = [layer for layer in self.style_layers]
    content_outputs = [layer for layer in self.content_layers]
    model_outputs = style_outputs + content_outputs
    return vgg_layers(model_outputs)
```

This function builds the VGG model to extract features from the specified layers.

#### Computing the Loss
```python
def compute_loss(self, outputs):
    style_outputs = outputs[:self.num_style_layers]
    content_outputs = outputs[self.num_style_layers:]

    style_loss = tf.add_n([get_style_loss(style_output, target) for style_output, target in zip(style_outputs, self.style_targets)])
    style_loss *= 1.0 / self.num_style_layers

    content_loss = tf.add_n([get_content_loss(content_output, target) for content_output, target in zip(content_outputs, self.content_targets)])
    content_loss *= 1.0 / self.num_content_layers

    loss = style_loss + content_loss
    return loss
```

This function computes the total loss as a weighted sum of the style loss and content loss.

#### Running the Optimization
```python
def run_style_transfer(self, num_iterations=1000, content_weight=1e3, style_weight=1e-2):
    self.content_image = load_and_process_img(self.content_path)
    self.style

_image = load_and_process_img(self.style_path)

    self.model = self.build_model()

    style_features = self.model(self.style_image)
    content_features = self.model(self.content_image)

    self.style_targets = [gram_matrix(style_feature) for style_feature in style_features[:self.num_style_layers]]
    self.content_targets = content_features[self.num_content_layers:]

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
        grads, all_loss = self.compute_grads(cfg)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, -1.0, 1.0)
        init_image.assign(clipped)

        if loss < best_loss:
            best_loss = loss
            best_img = deprocess_img(init_image.numpy())

        if i % 100 == 0:
            print(f"Iteration: {i}, Loss: {loss}")

    return best_img, best_loss
```

This function runs the style transfer optimization, updating the generated image to minimize the total loss over a specified number of iterations.

## 5. Running the Script
The `main.py` script is the entry point for running the Neural Style Transfer.

```python
import argparse
from nst import NeuralStyleTransfer
from utils import imshow

def main(args):
    nst = NeuralStyleTransfer(args.content_image, args.style_image)
    best_img, best_loss = nst.run_style_transfer(num_iterations=args.iterations)
    imshow(best_img, 'Output Image')
    best_img.save(args.output_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Style Transfer')
    parser.add_argument('content_image', type=str, help='Path to content image')
    parser.add_argument('style_image', type=str, help='Path to style image')
    parser.add_argument('output_image', type=str, help='Path to save the output image')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations for style transfer')
    args = parser.parse_args()
    main(args)
```

This script uses `argparse` to handle command-line arguments for specifying the paths to the content and style images, the output image path, and the number of iterations for the style transfer. It then creates an instance of `NeuralStyleTransfer`, runs the style transfer, and saves the generated image.

## 6. References
- Gatys, L.A., Ecker, A.S., & Bethge, M. (2016). Image Style Transfer Using Convolutional Neural Networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
- TensorFlow Tutorials: Neural Style Transfer. (https://www.tensorflow.org/tutorials/generative/style_transfer)

 