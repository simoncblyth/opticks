#!/usr/bin/env python
"""

https://docs.sympy.org/latest/modules/functions/elementary.html#piecewise

Hmm seems that sympy doesnt like a mix of symbolic and floating point 
maybe better to use 



"""
import logging
log = logging.getLogger(__name__)
import numpy as np
import matplotlib.pyplot as plt 
import scipy.integrate as integrate

if __name__ == '__main__':

    ri = np.array([
           [ 1.55 ,  1.478],
           [ 1.795,  1.48 ],
           [ 2.105,  1.484],
           [ 2.271,  1.486],
           [ 2.551,  1.492],
           [ 2.845,  1.496],
           [ 3.064,  1.499],
           [ 4.133,  1.526],
           [ 6.2  ,  1.619],
           [ 6.526,  1.618],
           [ 6.889,  1.527],
           [ 7.294,  1.554],
           [ 7.75 ,  1.793],
           [ 8.267,  1.783],
           [ 8.857,  1.664],
           [ 9.538,  1.554],
           [10.33 ,  1.454],
           [15.5  ,  1.454]
          ])


    en = ri[:,0]
    bi = 1. 
    ct = bi/ri[:,1]
    s2 = (1.-ct)*(1.+ct)

    s2i = integrate.cumtrapz( s2, en, initial=0. )  # initial keeps s2i shape same as s2 and en 
    s2i_0 = integrate.cumtrapz( s2, en )  

    fig, ax = plt.subplots(figsize=[12.8, 7.2])
    ax.plot( en, s2i, label="s2i" )
    ax.scatter( en, s2i, label="s2i" )

    ax.plot( en[:-1], s2i_0, label="s2i_0" )
 
    
    ax.legend()
    fig.show()




