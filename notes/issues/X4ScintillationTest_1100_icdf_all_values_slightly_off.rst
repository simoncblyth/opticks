X4ScintillationTest_1100_icdf_all_values_slightly_off
=======================================================

1100 beta fails now stand at::

    FAILS:  5   / 492   :  Thu Sep 23 18:54:04 2021   
      30 /31  Test #30 : ExtG4Test.X4ScintillationTest                 Child aborted***Exception:     0.55   
      3  /45  Test #3  : CFG4Test.CTestDetectorTest                    Child aborted***Exception:     4.72   
      5  /45  Test #5  : CFG4Test.CGDMLDetectorTest                    Child aborted***Exception:     4.62   
      7  /45  Test #7  : CFG4Test.CGeometryTest                        Child aborted***Exception:     5.04   
      27 /45  Test #27 : CFG4Test.CInterpolationTest                   Child aborted***Exception:     4.76   


X4ScintillationTest
-----------------------

No problem with 1042, O:: 

    x4 ; ipython -i tests/X4ScintillationTest.py 

    O[blyth@localhost extg4]$ ipython -i tests/X4ScintillationTest.py 
    Python 2.7.5 (default, Nov 16 2020, 22:23:17) 
    Type "copyright", "credits" or "license" for more information.

    IPython 3.2.1 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.
    INFO:__main__:icdf_jspath:/tmp/blyth/opticks/X4ScintillationTest/g4icdf_auto.json
    INFO:__main__: num_bins : 4096 
    INFO:__main__: edge : 0.05 
    INFO:__main__: hd_factor : 20 
    INFO:__main__: name : LS 
    INFO:__main__: creator : X4Scintillation::CreateGeant4InterpolatedInverseCDF 
    INFO:__main__:icdf_compare
    a:(3, 4096) a.min    200.118 a.max    799.898
    b.(3, 4096) b.min    200.118 b.max    799.898
    ab:(3, 4096) ab.min          0 ab.max          0

    In [1]: a
    Out[1]: 
    array([[ 799.89798414,  785.89756342,  772.37880425, ...,  208.95408887,
             205.87260891,  202.8806943 ],
           [ 799.89798414,  799.18612661,  798.47553497, ...,  485.01049867,
             485.004202  ,  484.99794481],
           [ 391.46193947,  391.46039661,  391.45885375, ...,  200.40510648,
             200.26136376,  200.11782709]])

    In [2]: ab
    Out[2]: 
    array([[ 0.,  0.,  0., ...,  0.,  0.,  0.],
           [ 0.,  0.,  0., ...,  0.,  0.,  0.],
           [ 0.,  0.,  0., ...,  0.,  0.,  0.]])


With 1100 using same geoache as O, not just a few values off, all values are off at 1e-5/1e-4 level::

    x4 ; ipython -i tests/X4ScintillationTest.py 

    NFO:__main__: num_bins : 4096 
    INFO:__main__:icdf_compare
    a:(3, 4096) a.min    200.118 a.max    799.898
    b.(3, 4096) b.min    200.118 b.max    799.898
    ab:(3, 4096) ab.min 1.7579e-05 ab.max 7.02658e-05

    In [1]: a
    Out[1]: 
    array([[799.89798414, 785.89756342, 772.37880425, ..., 208.95408887,
            205.87260891, 202.8806943 ],
           [799.89798414, 799.18612661, 798.47553497, ..., 485.01049867,
            485.004202  , 484.99794481],
           [391.46193947, 391.46039661, 391.45885375, ..., 200.40510648,
            200.26136376, 200.11782709]])

    In [2]: ab
    Out[2]: 
    array([[7.02658095e-05, 6.90359639e-05, 6.78484295e-05, ...,
            1.83552509e-05, 1.80845630e-05, 1.78217429e-05],
           [7.02658095e-05, 7.02032775e-05, 7.01408566e-05, ...,
            4.26050020e-05, 4.26044490e-05, 4.26038993e-05],
           [3.43873726e-05, 3.43872371e-05, 3.43871016e-05, ...,
            1.76042787e-05, 1.75916518e-05, 1.75790431e-05]])

    In [3]: ab.shape
    Out[3]: (3, 4096)


Compare the icdf directly with each other, not with cache::

    In [2]: import numpy as np

    In [3]: a = np.load("/tmp/simon/opticks/X4ScintillationTest/g4icdf_manual.npy")

    In [4]: b = np.load("/tmp/blyth/opticks/X4ScintillationTest/g4icdf_manual.npy")

    In [5]: ab = np.abs(a - b )

    In [6]: a
    Out[6]: 
    array([[[ 799.89805441],
            [ 785.89763246],
            [ 772.3788721 ],
            ..., 
            [ 208.95410723],
            [ 205.872627  ],
            [ 202.88071212]],

           [[ 799.89805441],
            [ 799.18619681],
            [ 798.47560511],
            ..., 
            [ 485.01054128],
            [ 485.00424461],
            [ 484.99798742]],

           [[ 391.46197386],
            [ 391.46043099],
            [ 391.45888814],
            ..., 
            [ 200.40512408],
            [ 200.26138135],
            [ 200.11784467]]])

    In [7]: ab.min()
    Out[7]: 1.7579043060322874e-05

    In [8]: ab.max()
    Out[8]: 7.026580954061501e-05
        


Huh, comparing in energy shows no change::

    (base) [simon@localhost extg4]$ ipython 
    Python 3.7.7 (default, May  7 2020, 21:25:33) 
    Type 'copyright', 'credits' or 'license' for more information
    IPython 7.18.1 -- An enhanced Interactive Python. Type '?' for help.

    In [1]: import numpy as np

    In [2]: a = np.load("/tmp/simon/opticks/X4ScintillationTest/g4icdf_energy_manual.npy")

    In [3]: b = np.load("/tmp/blyth/opticks/X4ScintillationTest/g4icdf_energy_manual.npy")

    In [4]: ab = np.abs(a-b)

    In [5]: ab.min()
    Out[5]: 0.0

    In [6]: ab.max()
    Out[6]: 0.0



Compare the constants, very small change in h_Planck::


    In [2]: import numpy as np

    In [3]: a = np.load("/tmp/simon/opticks/X4PhysicalConstantsTest/1100.npy")

    In [4]: b = np.load("/tmp/blyth/opticks/X4PhysicalConstantsTest/1042.npy")

    In [5]: a
    Out[5]: 
    array([4.13566770e-12, 2.99792458e+02, 1.23984198e-09, 1.23984198e-03,
           1.00000000e-06])

    In [6]: b
    Out[6]: 
    array([4.13566733e-12, 2.99792458e+02, 1.23984188e-09, 1.23984188e-03,
           1.00000000e-06])

    In [7]: a-b
    Out[7]: 
    array([3.63291343e-19, 0.00000000e+00, 1.08912005e-16, 1.08912005e-10,
           0.00000000e+00])



Compare the integrals, they match exactly::

    In [1]: import numpy as np

    In [2]: a = np.load("/tmp/simon/opticks/X4ScintillationTest/ScintillatorIntegral.npy")

    In [3]: b = np.load("/tmp/blyth/opticks/X4ScintillationTest/ScintillatorIntegral.npy")


    In [9]: ab = np.abs(a - b )

    In [10]: ab.min()
    Out[10]: 0.0

    In [11]: ab.max()
    Out[11]: 0.0


Compare metadata on the ScintillatorIntegral, get exact match::

    In [1]: import numpy as np

    In [2]: a = np.load("/tmp/simon/opticks/X4ScintillationTest/meta.npy")

    In [3]: b = np.load("/tmp/blyth/opticks/X4ScintillationTest/meta.npy")

    In [4]: a
    Out[4]: 
    array([  1.55000000e-05,   1.55000000e-06,   0.00000000e+00,
             4.12705877e-07,   0.00000000e+00])

    In [5]: b
    Out[5]: 
    array([  1.55000000e-05,   1.55000000e-06,   0.00000000e+00,
             4.12705877e-07,   0.00000000e+00])

    In [6]: a - b
    Out[6]: array([ 0.,  0.,  0.,  0.,  0.])

    In [7]: ab = np.abs(a-b)

    In [8]: ab.max()
    Out[8]: 0.0

    In [9]: ab.min()
    Out[9]: 0.0





