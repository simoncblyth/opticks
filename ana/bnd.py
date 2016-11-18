#!/usr/bin/env python
"""
::

    In [13]: m0.data.shape
    Out[13]: (38, 2, 39, 4)   # 38 materials, 2 groups, 39 wavelengths, 4 qwns in each group : 38 mats include 2 added ones: GlassSchottF2, MainH2OHale -> 36 standard ones

        ## huh m0.names shows that GlassSchottF2 appears in the middle, not at the tail ???  MainH2OHale is last 

    In [14]: b0.data.shape
    Out[14]: (123, 4, 2, 39, 4)  # 123 bnds, 4 matsur, 2 groups, 39 wavelengths, 4 qwns in each group

    In [15]: s0.data.shape       # 49 surs include 4 added "perfect" ones that dont appear in bnds, so 49-4 = 44 standard ones
    Out[15]: (48, 2, 39, 4)

    In [17]: len(b0.bnd.surs)   # includes 1 blank, so 44 surs occuring in bnd
    Out[17]: 45

    In [18]: len(b0.bnd.mats)   
    Out[18]: 36


    In [55]: np.where( np.logical_or( rel < -0.02, rel > 0.02 ))   ## 221/219168 off by more than 2%
    Out[55]: 
    (array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,
            1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,
            2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
            3,  4,  4,  4,  4,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8, 13, 13, 13, 13, 27, 27, 27, 27, 27, 27, 27,
           27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27]),
     array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),   ## all group zero, ie not GROUPVEL
     array([141, 142, 143, 144, 145, 145, 146, 146, 147, 147, 148, 148, 149, 149, 150, 150, 151, 152, 153, 154, 155, 288, 323, 324, 325, 326, 327, 328, 329, 330, 341, 342, 343, 344, 345, 346, 347, 348,
           349, 350, 351, 653, 654, 655, 658, 659, 141, 142, 143, 144, 145, 145, 146, 146, 147, 147, 148, 148, 149, 149, 150, 150, 151, 152, 153, 154, 155, 288, 323, 324, 325, 326, 327, 328, 329, 330,
           341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 653, 654, 655, 658, 659, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 141, 142, 143, 144, 145, 145, 146,
           146, 147, 147, 148, 148, 149, 149, 150, 150, 151, 152, 153, 154, 155, 145, 146, 147, 148, 149, 150, 224, 225, 226, 228, 229, 263, 264, 265, 268, 269, 308, 733, 734, 738, 739, 124, 125, 126,
           127, 128, 129, 130, 131, 124, 125, 126, 127, 128, 129, 130, 131, 124, 125, 126, 127, 128, 129, 130, 131, 124, 125, 126, 127, 128, 129, 130, 131, 733, 734, 738, 739, 121, 122, 123, 124, 125,
           126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 141, 142, 143, 144, 145, 145, 146, 146, 147, 147, 148, 148, 149, 149, 150, 150, 151, 152, 153, 154, 155]),
     array([1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1,
           2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1]))
                   ## never 0, refractive index, mostly absorption and scattering lengths 


Material property min/max relative interpolation differences::

                                                  RINDEX                   ABSLEN                 RAYLEIGH                 REEMPROB                 GROUPVEL  
 0                      GdDopedLS      -0.0048     0.0053       -0.0096     0.0821        0.0000     0.0237       -0.0423     0.0032       -0.0125     0.0065  
 1             LiquidScintillator      -0.0048     0.0053       -0.0100     0.0821        0.0000     0.0237       -0.0423     0.0032       -0.0125     0.0065  
 2                        Acrylic      -0.0046     0.0053        0.0000     0.0968        0.0000     0.0237        0.0000     0.0000       -0.0123     0.0064  
 3                     MineralOil      -0.0046     0.0053       -0.0083     0.0232        0.0000     0.0237        0.0000     0.0000       -0.0123     0.0063  
 4                       Bialkali       0.0000     0.0000       -0.0396     0.0017        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
 5                       IwsWater      -0.0001     0.0000       -0.0084     0.0254        0.0000     0.0000        0.0000     0.0000       -0.0006     0.0005  
 6                          Water      -0.0001     0.0000       -0.0084     0.0254        0.0000     0.0000        0.0000     0.0000       -0.0006     0.0005  
 7                      DeadWater      -0.0001     0.0000       -0.0084     0.0254        0.0000     0.0000        0.0000     0.0000       -0.0006     0.0005  
 8                       OwsWater      -0.0001     0.0000       -0.0084     0.0254        0.0000     0.0000        0.0000     0.0000       -0.0006     0.0005  
 9                            ESR       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
10                   OpaqueVacuum       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
11                           Rock       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
12                         Vacuum       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
13                          Pyrex       0.0000     0.0000       -0.0396     0.0017        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
14                            Air       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
15                  GlassSchottF2       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
16                            PPE       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
17                      Aluminium       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
18          ADTableStainlessSteel       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
19                           Foam       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
20                       Nitrogen      -0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000       -0.0000     0.0000  
21                    NitrogenGas       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
22                          Nylon       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
23                            PVC       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
24                          Tyvek       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
25                       Bakelite       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
26                         MixGas       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
27                           Iron      -0.0046     0.0053        0.0000     0.0968        0.0000     0.0237        0.0000     0.0000       -0.0123     0.0064  
28                         Teflon       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
29             UnstStainlessSteel       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
30                            BPE       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
31                          Ge_68       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
32                          Co_60       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
33                           C_13       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
34                         Silver       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
35                        RadRock       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  


"""
import os, logging, numpy as np
from collections import OrderedDict as odict
log = logging.getLogger(__name__)

from opticks.ana.base import opticks_main 
from opticks.ana.proplib import PropLib, Bnd

idp_ = lambda _:os.path.expandvars("$IDPATH/%s" % _ )



if __name__ == '__main__':
    ok = opticks_main()
    
    # from old geocache without groupvel setup
    m0 = PropLib("GMaterialLib") 
    s0 = PropLib("GSurfaceLib") 
    b0 = PropLib("GBndLib") 

    # persisted postcache modified bnd buffer including groupvel calc, identity wavelengths 20nm steps
    b1 = PropLib("GBndLib", data="$TMP/InterpolationTest/CInterpolationTest_identity.npy" )  
    b2 = PropLib("GBndLib", data="$TMP/InterpolationTest/OInterpolationTest_identity.npy" )  
    assert np.allclose( b1.data, b2.data ) # after unset alignment

    # persisted postcache groupvel calc, interpol wavelengths 1nm steps
    i1 = PropLib("GBndLib", data="$TMP/InterpolationTest/CInterpolationTest_interpol.npy" ) 
    i2 = PropLib("GBndLib", data="$TMP/InterpolationTest/OInterpolationTest_interpol.npy" ) 

    assert np.allclose( i1.data[:,:,:,::20,:], i2.data[:,:,:,::20,:] ) == True   # plucking identity wavelengths from the interpol ones
    assert np.allclose( i1.data[:,:,:,::20,:], b1.data  )
    assert np.allclose( i2.data[:,:,:,::20,:], b2.data  )


    # collapse material duplication for simpler comparisons
    order = m0.names
    b0m = b0.as_mlib(order=order)
    b1m = b1.as_mlib(order=order)
    b2m = b2.as_mlib(order=order)
    i1m = i1.as_mlib(order=order)
    i2m = i2.as_mlib(order=order)

    assert np.all( i1m.names == i2m.names ) 
    matnames = i1m.names
    assert np.all( b0m.data == m0.data[~np.logical_or(m0.names == 'GlassSchottF2', m0.names == 'MainH2OHale')] )
    assert np.all( b1m.data == b2m.data )   ## identity match

    assert np.all( i1m.data[:,:,::20,:] == i2m.data[:,:,::20,:] )   ## on the raster, where there is no interpolation to do, get perfect match


    avg = (i1m.data + i2m.data)/2.0
    dif = i1m.data - i2m.data                 # absolute difference

    # signed relative difference of 36*2*761*4 = 219168 property values 
    rel = np.where( np.logical_or(avg < 1e-6, dif == 0), 0, dif/avg )

   
    rd = np.zeros( (len(rel), 2, 4, 2), dtype=np.float32 ) 
    rd[:,:,:,0] = np.amin(rel, axis=2) 
    rd[:,:,:,1] = np.amax(rel, axis=2) 



    print "".join(["%2s %30s " % ("","")] + map(lambda plab:"  %21s  " % plab, PropLib.M_LABELS ))
    for i in range(len(rd)):
        labl = ["%2d %30s " % ( i, matnames[i] )]
        rnge =  map(lambda mimx:"  %10.4f %10.4f  " % ( float(mimx[0]), float(mimx[1])) , rd[i].reshape(-1,2)[:5] )
        print "".join(labl + rnge)




