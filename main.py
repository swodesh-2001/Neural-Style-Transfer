import argparse
from nst import NeuralStyleTransfer
from utils import imshow
import tensorflow as tf

# Configure TensorFlow to use GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def main(args):
    nst = NeuralStyleTransfer(args.content_image, args.style_image)
    best_img, best_loss = nst.run_style_transfer(num_iterations=args.iterations)
    imshow(best_img, 'Output Image')
    best_img.save(args.output_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural Style Transfer')
    parser.add_argument('--content_image', type=str, help='Path to content image')
    parser.add_argument('--style_image', type=str, help='Path to style image')
    parser.add_argument('--output_image', type=str, help='Path to save the output image')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations for style transfer')
    args = parser.parse_args()
    main(args)
