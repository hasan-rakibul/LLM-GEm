import matplotlib.pyplot as plt
import scienceplots

import torch
import numpy as np

def plot(x, y, y2=None, xlabel=None, ylabel=None, legend=[], save=False, filename=None):
    """Plot data points"""
    plt.style.use(['science'])
    fig, ax = plt.subplots(1,1)
    
    ax.plot(x,y)
    if y2 is not None:
        ax.plot(x, y2)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(legend)
    
    if save:
        plt.savefig(fname=filename+'.pdf', format = 'pdf', bbox_inches='tight')
        print(f"Saved as {filename}.pdf")
        
    fig.show()


def get_device(device_id=0):
    return torch.device("cuda", device_id) if torch.cuda.is_available() else torch.device("cpu")

def set_all_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)