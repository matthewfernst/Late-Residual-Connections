import LateResidualConnections as lrc
import argparse

parser = argparse.ArgumentParser(description='Run Late Residual Neural Network')
parser.add_argument('-w','--width', type=int, default=2, help='Width of the network')
parser.add_argument('-d', '--depths', type=int, nargs='+', default=[2, 5, 10, 20, 25, 30], help='Depths of the network (list format)')
parser.add_argument('-lr', '--learning_rates', type=int, nargs='+', default=[0.01, 0.001, 0.1], help='Learning rates (list format)')
parser.add_argument('-opt', '--optimizers', type=str, nargs='+', default=['Adam', 'SGD'], help='Optimizers (list format)')
parser.add_argument('-e', '--epochs', type=int, default=3000, help='Number of epochs')

if __name__ == "__main__":
    args = vars(parser.parse_args())
    print(args)
    # lrc.run(args['width'], args['depths'], args['learning_rates'], args['optimizers'], args['epochs'])
