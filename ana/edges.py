#!/usr/bin/env python 

import logging
log = logging.getLogger(__name__)
import numpy as np
import matplotlib.pyplot as plt 

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


def divide_bins( e, mul ):
    """
    :param e: 1d array of monotonic edges
    :param mul: integer multiplicity with which to divide edges, eg 2 or 3 splits each edge into 2 or 3 etc.. 
    :return ee: array with extra edges obtained by splitting the input edges  


         +--------+--------+     3 values, 2 bins

         +----+---+---+----+     5 values, 4 bins    (mul 2)

         +--+-+-+-+-+-+--+-+     9 values, 8 bins     


    """
    nb = len(e)-1           # number of bins is one less than number of values 
    nbb = nb*mul            # bins multiply  
    ee = np.zeros(nbb+1)    # add one to to give number of values 
 
    print(" divide_bins mul %d len(e) %d len(ee) %d " % ( mul, len(e), len(ee) )) 

    for i in range(len(e)-1):
        a = np.linspace( e[i], e[i+1], 1+mul )
        i0 = i*mul
        i1 = (i+1)*mul
        print( " %2d %7.4f %7.4f i0 %2d i1 %2d     %s " % (i, e[i], e[i+1], i0, i1, a ) )
        ee[i0:i1+1] = a
    pass
    if mul == 1: assert np.all( ee == e ) 

    non_monotonic = np.any(ee[:-1] > ee[1:])
    if non_monotonic:
        log.error("non_monotonic")
        print(ee)
    pass
    return ee 



def edges_plot(e):
    fig, ax = plt.subplots(figsize=[12.8, 7.2])
    for m in range(1,10):
        ee = divide_bins(e, mul=m)
        yy = np.repeat( m, len(ee) )
        ax.scatter( ee, yy, label="mul %d " % m ) 
    pass
    ax.legend()
    fig.show()     



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    e = ri[:,0]
    #ee = divide_bins(e, mul=2 )
    edges_plot(e) 



