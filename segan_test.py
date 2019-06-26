import h5py
import numpy as np
from segan.segan.models import *

class ArgParser(object):

    def __init__(self, args):
        for k, v in args.items():
            setattr(self, k, v)

with open("./segan/ckpt_segan+/train.opts", 'r') as cfg_f:
    args = ArgParser(json.load(cfg_f))
    print('Loaded train config: ')
    print(json.dumps(vars(args), indent=2))

segan = SEGAN(args)
segan.G.load_pretrained("./segan/ckpt_segan+/segan+_generator.ckpt", True)
segan.G.eval()

def segan_hr(X):
    X_hr = np.empty(X.shape)
    for wav in X:
        pwav = torch.FloatTensor(wav).view(1, 1, -1)
        g_wav, g_c = segan.generate(pwav)
        np.append(X_hr, g_wav)

    return X_hr
