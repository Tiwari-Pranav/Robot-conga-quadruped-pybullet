# Configuration file for ConvoyControl
import numpy as np

# Functions
def xt(t):
    return np.square(t)*1e-1

def yt(t):
    return t 

def d_xt(t):
    return t *2e-1

def d_yt(t):
    return 1

# Time computation function
def compute_t(num_bots):
    return 1.5 * np.arange(num_bots - 1, -1, -1) + 5.

# Other parameters
num_bots = 4
linear_vel = 0.5
y = 2.5
a = 4.5
b = 1.5
dt = 7e-3
angle_change = np.pi/6