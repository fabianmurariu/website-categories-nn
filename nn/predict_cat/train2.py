import sys, getopt
from os import path

from . import train as tr

def main(features_path, model_path, epochs):
    pass


if __name__ == "__main__":
    print("ARGS", sys.argv)
    try:
        opts, args = getopt.getopt(sys.argv[1:], "f", ["features_path=", "epochs=", "model_out="])
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    opts_dict = {k: v for k, v in opts}

    features_path = path.abspath(path.expanduser(opts_dict['--features_path']))
    model_path = path.abspath(path.expanduser(opts_dict['--model_out']))
    epochs = int(opts_dict.get('--epochs', 50))
    main(features_path=features_path, model_path=model_path, epochs=epochs)
