#!/usr/bin/env python

import numpy as np

def rotateUz(d, u):
    """
    :param d:  array of vectors with shape (n,3)
    :param u:  array of single vector with shape (3,)
    :return d1: array of vectors with shape (n,3)

    NumPy translation of smath.h rotateUz::

        inline SMATH_METHOD void smath::rotateUz(float3& d, const float3& u )
        {
            float up = u.x*u.x + u.y*u.y ;
            if (up>0.f)
            {
                up = sqrt(up);
                float px = d.x ;
                float py = d.y ;
                float pz = d.z ;
                d.x = (u.x*u.z*px - u.y*py)/up + u.x*pz;
                d.y = (u.y*u.z*px + u.x*py)/up + u.y*pz;
                d.z =    -up*px +                u.z*pz;
            }
            else if (u.z < 0.f )
            {
                d.x = -d.x;
                d.z = -d.z;
            }
        }

    """
    assert u.shape == (3,)
    assert d.shape == (len(d),3)
    d1 = np.zeros( (len(d),3 ), dtype=d.dtype )

    up =  u[0]*u[0] + u[1]*u[1]
    if up > 0.:
       up = np.sqrt(up)
       px = d[:,0]
       py = d[:,1]
       pz = d[:,2]
       d1[:,0] = (u[0]*u[2]*px - u[1]*py)/up + u[0]*pz
       d1[:,1] = (u[1]*u[2]*px + u[0]*py)/up + u[1]*pz
       d1[:,2] = -up*px + u[2]*pz
    elif u[2] < 0.:
       d1[:,0] = -d[:,0]
       d1[:,1] =  d[:,1]  # diff as returning not inplace changing
       d1[:,2] = -d[:,2]
    pass
    return d1


def rotateUz_(d, u):
    """
    :param d:  array of single vector with shape (3,)
    :param u:  array of vectors with shape (n,3)
    :return d1: array of vectors with shape (n,3)

    THIS NEEDS MORE TESTING
    """
    assert u.shape == (len(u),3)
    assert d.shape == (3,)
    d1 = np.zeros( (len(u),3 ), dtype=d.dtype )

    _up =  u[:,0]*u[:,0] + u[:,1]*u[:,1]
    wp = np.where( _up > 0. )
    wn = np.where( np.logical_and( _up <= 0., u[:,2] < 0. ))

    up = np.zeros( (len(u),), dtype=d.dtype )
    up[wp] = np.sqrt(_up)
    px = d[0]
    py = d[1]
    pz = d[2]

    d1[wp,0] = (u[wp,0]*u[wp,2]*px - u[wp,1]*py)/up + u[wp,0]*pz
    d1[wp,1] = (u[wp,1]*u[wp,2]*px + u[wp,0]*py)/up + u[wp,1]*pz
    d1[wp,2] = -up[wp]*px + u[wp,2]*pz

    d1[wn,0] = -d[0]
    d1[wn,1] =  d[1]
    d1[wn,2] = -d[2]

    return d1


def test_rotateUz():
    s = np.sqrt(0.5, dtype=np.float64)
    u = np.array([s,0,-s], dtype=s.dtype)
    d = np.array([[1,0,0],[s,s,0],[0,1,0],[-s,s,0],[-1,0,0],[-s,-s,0],[0,-1,0],[s,-s,0],[1,0,0]], dtype=s.dtype )
    d1 = rotateUz( d, u )

    print("u\n",u)
    print("d\n",d)
    print("d1\n",d1)



if __name__ == '__main__':

    s = np.sqrt(0.5, dtype=np.float64)
    u = np.array([[1,0,0],[s,s,0],[0,1,0],[-s,s,0],[-1,0,0],[-s,-s,0],[0,-1,0],[s,-s,0],[1,0,0]], dtype=s.dtype )
    d = np.array([s,0,-s], dtype=s.dtype)
    d1 = rotateUz_( d, u )

    print("u\n",u)
    print("d\n",d)
    print("d1\n",d1)


