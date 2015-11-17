Torch Stratification
======================

Refs
-----

* http://paulbourke.net/miscellaneous/aliasing/

::

    Since the resultant colour of each pixel is based upon one infinitely small
    sample taken within the centre of each pixel and because pixels occur at
    regular intervals frequency based aliasing problems often arise. Aliasing
    refers to the inclusion of characteristics or artifacts in an image that could
    have come from more than one scene description.


Issue
------

100mm radius beam head on to PMT.  

* Observe clear dinner plate (Fresnel lens) 
  looking stratification in the beam, prior to hitting PMT, 
  categories "BT SA" and "BT SD" show it clearest.

* Moving the PMT out of the way with "slice=0:0" gets rid of the stratification.


.. image:: //env/graphics/ggeoview/issues/statification.png
   :width: 900px
   :align: center


::

    ggv-pmt-test(){
       type $FUNCNAME

       local torch_config=(
                     type=disc
                     photons=500000
                     frame=1
                     source=0,0,300
                     target=0,0,0
                     radius=25
                     zenithazimuth=0,1,0,1
                     material=Vacuum
                   )

       local test_config=(
                     mode=PmtInBox
                     boundary=Rock//perfectAbsorbSurface/MineralOil
                     dimensions=300,0,0,0
                     shape=B
                     analytic=1
                       ) 

       ggv \
           --test --testconfig "$(join _ ${test_config[@]})" \
           --torch --torchconfig "$(join _ ${torch_config[@]})" \
           --animtimemax 10 \
           --eye 0.5,0.5,0.0 \
           $* 

    }


::

    [2015-Nov-17 14:55:14.112908]:info: 
        0    442214     0.884                      8cd                               TO BT SA 
        1     52171     0.104                      7cd                               TO BT SD 
        2      3613     0.007                       4d                                  TO AB 
        3      1040     0.002                      86d                               TO SC SA 
        4       743     0.001                      4cd                               TO BT AB 
        5       164     0.000                     8c6d                            TO SC BT SA 
        6        25     0.000                     7c6d                            TO SC BT SD 
        7        11     0.000                      46d                               TO SC AB 
        8         8     0.000                      8bd                               TO BR SA 
        9         7     0.000                    8cc6d                         TO SC BT BT SA 
       10         2     0.000                     8b6d                            TO SC BR SA 
       11         1     0.000                     4c6d                            TO SC BT AB 
       12         1     0.000                     866d                            TO SC SC SA 
      TOT    500000

    [2015-Nov-17 14:55:14.113700]:info: App::indexSequence m_seqmat
    [2015-Nov-17 14:55:14.113868]:info: 
        0    495128     0.990                      ee4                               MO Py Py 
        1      3613     0.007                       44                                  MO MO 
        2      1059     0.002                      444                               MO MO MO 
        3       190     0.000                     ee44                            MO MO Py Py 
        4         7     0.000                    44e44                         MO MO Py MO MO 
        5         3     0.000                     4444                            MO MO MO MO 
      TOT    500000




Using tracer mode for fast turnaround vary the slice to find just the front part of PMT, 
then run without tracer for propagation::

    ggv-;ggv-pmt-test --tracer

    #slice=2:3

    ggv-;ggv-pmt-test


See same stratification pattern with just the MO/Pyrex of very front face, just 
not quite as wide.  


Visualization Artifact Only ? 
---------------------------------

Plotting the z position of the intersect shows no stair stepping.


Compression Artifact ? Dont think so
---------------------------------------

Where does the position come from:

* the intersection point with sphere is calculated and than a linear interpolation 
  between the steps based on input time provides the position


::

    In [1]: np.load("OPropagatorF.npy")
    Out[1]: 
    array([[[   0.,    0.,    0.,  700.]],      # center extent domain  

           [[   0.,  200.,    7.,    0.]],

           [[  60.,  810.,   20.,  750.]]], dtype=float32)


::

   Compression extent is 700mm
   Front part of PMT radius of curvature 131mm


cu/photon.h::

   int inorm = __float2int_rn(32767.0f * (v - center)/extent ) ;    // linear scaling into -1.f:1.f * float(SHRT_MAX)


::

    [2015-Nov-17 16:16:41.757108]:info: OGeo::makeAnalyticGeometry partBuf (2,4,4) 

    (  0)       0.000       0.000       0.000     131.000 
    (  0)       0.000       0.000       0.000       0.000 
    (  0)     -84.540     -84.540     100.070       0.000 
    (  0)      84.540      84.540     131.000       0.000 
    (  1)       0.000       0.000       0.000     300.000 
    (  1)       0.000       0.000       0.000       0.000 
    (  1)    -300.003    -300.003    -300.003       0.000 
    (  1)     300.003     300.003     300.003       0.000 


Save the data::

   ggv-;ggv-pmt-test --save

::

    In [1]: run stratification.py
    -rw-r--r--  1 blyth  staff  32000080 Nov 17 16:56 /usr/local/env/dayabay/oxtorch/1.npy
    -rw-r--r--  1 blyth  staff  80000080 Nov 17 16:56 /usr/local/env/dayabay/rxtorch/1.npy
    -rw-r--r--  1 blyth  staff  8000080 Nov 17 16:56 /usr/local/env/dayabay/phtorch/1.npy

    In [2]: e.history_table()
                     8cd     345363 :                              TORCH BT SA 
                      8d     137769 :                                 TORCH SA 
                      4d       6276 :                                 TORCH AB 
                     4cd       6192 :                              TORCH BT AB 

    In [3]: s = Selection(e,"BT SA")  # select the most prolific category 

    In [8]: z = s.recpos(1)[:,2]      # z position of record index 1, ie PMT Pyrex intersection z 

    In [9]: z
    Out[9]: array([ 129.505,  112.687,  102.414, ...,  113.428,  119.691,  102.432], dtype=float32)

    In [10]: z.min()
    Out[10]: 100.07019

    In [12]: z.max()
    Out[12]: 130.99765

    In [14]: iz = s.rx[:,1,0,2]

    In [15]: z.shape
    Out[15]: (345363,)

    In [16]: iz.shape
    Out[16]: (345363,)

    In [17]: iz.min()
    Out[17]: 10930

    In [18]: iz.max()
    Out[18]: 14308


Huh extent now 300mm::

    In [24]: e.fdom
    Out[24]: 
    array([[[   0.,    0.,    0.,  300.]],

           [[   0.,  200.,   10.,    0.]],

           [[  60.,  810.,   20.,  750.]]], dtype=float32)

    In [25]: float(iz.min())/32767.0*300.
    Out[25]: 100.07019257179479

    In [26]: float(iz.max())/32767.0*300.
    Out[26]: 130.99765007477035


Sufficiently small seems unlikely to cause that much strat::

    In [29]: iz.max() - iz.min()
    Out[29]: 3378

    In [30]: 1./32767.0*300.
    Out[30]: 0.009155552842799158

Real histo of record data shows nothing unexpected::

    In [36]: plt.hist(z, bins=3379)
    Out[36]: 
    (array([  52.,   76.,   84., ...,  113.,  122.,   84.]),
     array([ 100.07 ,  100.079,  100.088, ...,  130.979,  130.988,  130.998]),
     <a list of 3379 Patch objects>)


