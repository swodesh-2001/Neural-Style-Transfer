import argparse
from neuraltransfer import NeuralStyleTransfer

def parse_args():
    parser = argparse.ArgumentParser(description='Neural Style Transfer')
    parser.add_argument('--content', type=str, required=True, help='Path to the content image')
    parser.add_argument('--style', type=str, required=True, help='Path to the style image')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output image')
    parser.add_argument('--iteration', type= int, required=True, help='Number of iterations')
    parser.add_argument('--img_size', type= int, required=True, help='Image size')
    return parser.parse_args()

def main():
    args = parse_args()
    nst = NeuralStyleTransfer(args.content, args.style, args.output, iteration= args.iteration , img_size= args.img_size)
    nst.generate()

if __name__ == '__main__':
    main()
