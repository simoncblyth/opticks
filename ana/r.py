#!/usr/bin/env python

import numpy as np

def make_record_cells(r):
    """ 
    :param r: step record array, eg with shape (1000, 10, 4, 4)
    :return cells: eg with shape (1000,10+1 )

    * indices of the cells reference record points in flattened record array 

    In [1]: cells
    Out[1]: 
    array([[ 5,  0,  1,  2,  3,  4],
           [ 5,  5,  6,  7,  8,  9],
           [ 5, 10, 11, 12, 13, 14],
           [ 5, 15, 16, 17, 18, 19],
           [ 5, 20, 21, 22, 23, 24],
           [ 5, 25, 26, 27, 28, 29],
           [ 5, 30, 31, 32, 33, 34],
           [ 5, 35, 36, 37, 38, 39]])

    """
    assert r.ndim == 4   
    assert r.shape[2:] == (4,4)
    num_pho, max_rec = r.shape[:2]
    cells = np.zeros( (num_pho,max_rec+1), dtype=np.int ) 
    offset = 0 
    for i in range(num_pho):
        cells[i,0] = max_rec
        cells[i,1:] = np.arange( 0, max_rec ) + offset 
        offset += max_rec 
    pass
    return cells 


