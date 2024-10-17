""" Make an interger-only circuit of the corresponding CIRCOM circuit.

Usage:
    circuit.py <circuit.json> <input.json> [-o <output>]
    circuit.py (-h | --help)

Options:
    -h --help                               Show this screen.
    -o <output> --output=<output>           Output directory [default: output].

"""

from docopt import docopt
import json

try:
    from keras2circom.util import *
except:
    import sys
    import os
    # add parent directory to sys.path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from keras2circom.util import *

def inference(input, circuit):
    out = input['in']
    output = {}
    output['in'] = out
    
    out, remainder = Conv2DInt(28, 28, 1, 4, 3, 1, 10**18, out, circuit['conv2d_weights'], circuit['conv2d_bias'])
    output['conv2d_out'] = out
    output['conv2d_remainder'] = remainder

    out, remainder = BatchNormalizationInt(26, 26, 4, 10**18, out, circuit['batch_normalization_a'], circuit['batch_normalization_b'])
    output['batch_normalization_out'] = out
    output['batch_normalization_remainder'] = remainder

    out = ReLUInt(26, 26, 4, out)
    output['activation_out'] = out

    out = MaxPooling2DInt(26, 26, 4, 2, 2, out)
    output['max_pooling2d_out'] = out

    out, remainder = Conv2DInt(13, 13, 4, 8, 3, 2, 10**18, out, circuit['conv2d_1_weights'], circuit['conv2d_1_bias'])
    output['conv2d_1_out'] = out
    output['conv2d_1_remainder'] = remainder

    out = ReLUInt(6, 6, 8, out)
    output['re_lu_out'] = out

    out, remainder = AveragePooling2DInt(6, 6, 8, 2, 2, out)
    output['average_pooling2d_out'] = out
    output['average_pooling2d_remainder'] = remainder

    out = Flatten2DInt(3, 3, 8, out)
    output['flatten_out'] = out

    out, remainder = DenseInt(72, 10, 10**18, out, circuit['dense_weights'], circuit['dense_bias'])
    output['dense_out'] = out
    output['dense_remainder'] = remainder

    out = ArgMaxInt(out)
    output['dense_softmax_out'] = out


    return out, output


def main():
    """ Main entry point of the app """
    args = docopt(__doc__)
    
    # parse input.json
    with open(args['<input.json>']) as f:
        input = json.load(f)
    
    # parse circuit.json
    with open(args['<circuit.json>']) as f:
        circuit = json.load(f)

    out, output = inference(input, circuit)

    # write output.json
    with open(args['--output'] + '/output.json', 'w') as f:
        json.dump(output, f)

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
