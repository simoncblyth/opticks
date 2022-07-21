#!/usr/bin/env python

import numpy as np


    
def pick_most_orthogonal_axis( a ):
    """
    The axis most orthogonal to a is the one
    with the smallest ordinate. 
    """
    if a[0] <= a[1] and a[0] <= a[3]:
        ax = np.array( [1,0,0,0] )
    elif a[1] <= a[0] and a[1] <= a[3]: 
        ax = np.array( [0,1,0,0] )
    elif a[2] <= a[0] and a[2] <= a[3]: 
        ax = np.array( [0,0,1,0] )
    else:
        assert 0 
    pass
    return ax

def make_rotation_matrix( a, b ):
    """
    :param a: unit vector
    :param b: unit vector
    :return m: matrix that rotates a to b  

    http://cs.brown.edu/research/pubs/pdfs/1999/Moller-1999-EBA.pdf
    "Efficiently Building a Matrix to Rotate One Vector To Another"
    Tomas Moller and John F Hughes 

    ~/opticks_refs/Build_Rotation_Matrix_vec2vec_Moller-1999-EBA.pdf

    Found this paper via thread: 
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d


           c + h vx vx     h vx vy - vz     h vx vz + vy  

           h vx vy + vz      c + h vy vy    h vy vz - vx 

           h vx vz - vy      h vy vz + vx    c + h vz vz    

    Where:: 

            c = a.b  

            h = (1 - c) /(1-c*c)

    This technique assumes the input vectors are normalized 
    but does no normalization itself. 


    When a and b are near parallel (or anti-parallel) 
    the absolute of the dot product is close to one so the
    basis for the rotation is poorly defined. 

    u = x - a
    v = x - b 

    """
    assert a.shape == (4,)
    assert b.shape == (4,)

    c = np.dot(a[:3],b[:3])

    rot = np.zeros((4,4))
    rot[3,3] = 1. 

    if np.abs(c) > 0.99:
        x = pick_most_orthogonal_axis(a)
        u = x[:3] - a[:3] 
        v = x[:3] - b[:3]

        uu = np.dot(u, u)
        vv = np.dot(v, v)
        uv = np.dot(u, v)

        for i in range(3):
            for j in range(3):
                delta_ij = 1. if i == j else 0.  
                rot[i,j] = delta_ij - 2.*u[i]*u[j]/uu -2.*v[i]*v[j]/vv + 4.*uv*v[i]*u[j]/(uu*vv)   
            pass
        pass
    else:
        vx,vy,vz = np.cross(a[:3],b[:3])  # cross product of a and b is perpendicular to both a and b 
        h = (1. - c)/(1. - c*c)
        rot[0,:3] = [c + h*vx*vx, h*vx*vy - vz,  h*vx*vz + vy] 
        rot[1,:3] = [h*vx*vy + vz, c + h*vy*vy,  h*vy*vz - vx]
        rot[2,:3] = [h*vx*vz - vy, h*vy*vz + vx, c + h*vz*vz ] 
    pass
    return rot
  



  
if __name__ == '__main__':

    DTYPE = np.float64
    X = np.array([1,0,0,0], dtype=DTYPE)
    Y = np.array([0,1,0,0], dtype=DTYPE)
    Z = np.array([0,0,1,0], dtype=DTYPE)
    O = np.array([0,0,0,1], dtype=DTYPE)

    a = Z
    b = -Z 

    #b = (X+Y)/np.sqrt(2.) 

    m = make_rotation_matrix( a, b )
    b2 = np.dot(m, a )
    assert np.allclose( b, b2 )


