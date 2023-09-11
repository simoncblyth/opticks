#pragma once


#if defined(__CUDACC__) || defined(__CUDABE__)
   #define SMATH_METHOD __device__
#else
   #define SMATH_METHOD 
#endif 


struct smath
{
    static constexpr float hc_eVnm = 1239.8418754200f ; // G4: h_Planck*c_light/(eV*nm) 

    SMATH_METHOD static void rotateUz(float3& d, const float3& u ); 
    SMATH_METHOD static int count_nibbles( unsigned long long ); 
    SMATH_METHOD static float erfcinvf(float u2); 
}; 


/**
smath::rotateUz
-----------------

This rotates the reference frame of a vector such that the original Z-axis will lie in the
direction of *u*. Many rotations would accomplish this; the one selected
uses *u* as its third column and is given by the below matrix.

The below CUDA implementation follows the CLHEP implementation used by Geant4::

     // geant4.10.00.p01/source/externals/clhep/src/ThreeVector.cc
     72 Hep3Vector & Hep3Vector::rotateUz(const Hep3Vector& NewUzVector) {
     73   // NewUzVector must be normalized !
     74 
     75   double u1 = NewUzVector.x();
     76   double u2 = NewUzVector.y();
     77   double u3 = NewUzVector.z();
     78   double up = u1*u1 + u2*u2;
     79 
     80   if (up>0) {
     81       up = std::sqrt(up);
     82       double px = dx,  py = dy,  pz = dz;
     83       dx = (u1*u3*px - u2*py)/up + u1*pz;
     84       dy = (u2*u3*px + u1*py)/up + u2*pz;
     85       dz =    -up*px +             u3*pz;
     86     }
     87   else if (u3 < 0.) { dx = -dx; dz = -dz; }      // phi=0  teta=pi
     88   else {};
     89   return *this;
     90 }

This implements rotation of (px,py,pz) vector into (dx,dy,dz) 
using the below rotation matrix, the columns of which must be 
orthogonal unit vectors.::

                |  u.x * u.z / up   -u.y / up    u.x  |        
        d  =    |  u.y * u.z / up   +u.x / up    u.y  |      p
                |   -up               0.         u.z  |      
    
Taking dot products between and within columns shows that to 
be the case for normalized u. See oxrap/rotateUz.h for the algebra. 

Special cases:

u = [0,0,1] (up=0., !u.z<0.) 
   does nothing, effectively identity matrix

u = [0,0,-1] (up=0., u.z<0. ) 
   flip x, and z which is a rotation of pi/2 about y 

               |   -1    0     0   |
      d =      |    0    1     0   |   p
               |    0    0    -1   |
           
**/

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

/**
smath::count_nibbles
---------------------

Refer to SBit::count_nibbles for explanation. 

**/
inline SMATH_METHOD int smath::count_nibbles(unsigned long long x)
{
    x |= x >> 1 ; 
    x |= x >> 2 ; 
    x &= 0x1111111111111111ull ; 
    x = (x + (x >> 4)) & 0xF0F0F0F0F0F0F0Full ; 
    unsigned long long count = (x * 0x101010101010101ull) >> 56 ; 
    return count ; 
}


#ifdef MOCK_CUDA
// this defines global erfcinvf function as standin for 
#include "s_mock_erfcinvf.h"
#endif

/**
smath::erfcinvf
----------------

Actually little need for this as CUDA already provides an erfcinvf function,
and for MOCK_CUDA a corresponding global is also defined based on 
using njuffa_erfcinvf.h.
However as globals tend to be difficult to find its convenient to 
include in smath.h for elucidatory+discovery purposes.

+-------------+--------------+
| domain      | erfcinvf(x)  |
+=============+==============+
|      x < 0  |  nan         |
+-------------+--------------+
|      x = 0  |   inf        |      
+-------------+--------------+
|      x = 1  |    0         |  
+-------------+--------------+
|      x = 2  |  -inf        |  
+-------------+--------------+

CAUTION: for matching with Geant4 the erfcinvf result is scaled 
by -sqrtf(2.f) with domain folded in half from 0->2 to 0->1
See Geant4/CLHEP classes::

    g4-cls RandGaussQ
    g4-cls G4MTRandGaussQ
    g4-cls G4OpBoundaryProcess

Geant4 sigma_alpha ground surface smears normal using angle from::

    alpha = G4RandGauss::shoot(0.0,sigma_alpha);  // (mean, stdDev) 

Tests while trying to do this on GPU::

    sysrap/tests/S4MTRandGaussQTest.sh
    sysrap/tests/erfcinvf_Test.sh
    sysrap/tests/njuffa_erfcinvf_test.sh
    sysrap/tests/smath_test.sh

**/

inline SMATH_METHOD float smath::erfcinvf(float u2) 
{
#if defined(__CUDACC__) || defined(__CUDABE__) || defined(MOCK_CUDA)
    return ::erfcinvf(u2) ; 
#else
    return 0.f ; 
#endif
}


