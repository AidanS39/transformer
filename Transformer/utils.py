import time
import argparse
import torch
import warnings

class EpochLog():
    def __init__(self, epoch):
        self.epoch = epoch
        self.batch_logs = list()
    def add_batch_log(self, batch, loss):
        self.batch_logs.append((batch, loss))

class TrainLog():
    def __init__(self):
        self.epoch_logs: list[EpochLog] = list()
        self.start_time = None
        self.end_time = None
    def add_epoch_log(self, epoch_log: EpochLog):
        self.epoch_logs.append(epoch_log)
    def start_timer(self):
        if self.start_time == None:
            self.start_time = time.time()
    def stop_timer(self):
        if self.end_time == None:
            self.end_time = time.time()

def get_arguments():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('--help', action='help', help='help menu')

    parser.add_argument("-m", "--d_model", type=int, default=1024, help="model dimension")
    parser.add_argument("-h", "--n_heads", type=int, default=8, help="number of heads")
    parser.add_argument("-l", "--n_layers", type=int, default=8, help="number of layers")
    parser.add_argument("-u", "--d_up", type=int, default=2048, help="up dimension")
    parser.add_argument("-e", "--n_epochs", type=int, default=5, help="number of epochs")

    args = parser.parse_args()

    print(f"D_MODEL {args.d_model}")
    print(f"N_HEADS {args.n_heads}")
    print(f"N_LAYERS {args.n_layers}")
    print(f"D_UP {args.d_up}")
    print(f"N_EPOCHS {args.n_epochs}")

    return args

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda") 
    else:
        warnings.warn("CUDA device not available, will use CPU instead.", UserWarning)
        return torch.device("cpu")