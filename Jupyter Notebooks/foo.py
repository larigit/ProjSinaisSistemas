import numpy as np
import matplotlib.pyplot as plt # Matplotlib

def foo(x):
    return -1j*((2j*np.pi*x*np.exp(2j*np.pi*x) - np.exp(2j*np.pi*x) + 1) / (2*np.pi*x**2))

result = []
for i in range(0,18):
    result.append(foo(i))

plt.plot(np.abs(result))
