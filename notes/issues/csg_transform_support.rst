CSG Transform Support
=========================


CSG Tree Objective
----------------------

* recall the CSG trees are intended to be small per-solid trees
  corresponding to shape definitions (ie not definitions of full scene geometry)



Transforms with Ray-trace and SDF
------------------------------------

To translate or rotate a surface modeled as an CSG tree, 
apply the inverse transformation to the point for SDFs or the ray for 
raytracing before doing the SDF distance calc or ray tracing intersection
calc.



Use higher level optix geometry transforms ?
-----------------------------------------------

Nope, I dont think this is possible as with boolean CSG need 
to apply different transforms to basis shapes underneath a single optix primitive.


Making a buffer of Matrix4x4 ?
-------------------------------

There is no RTformat for Matrix4x4 so would need 
use USER format buffer...


bringing over the gtransforms (ie compound transforms)
--------------------------------------------------------

* clearly want to do the matrix multiplication once
  CPU side

* hmm now that have moved bbox calc to GPU side, does it make sense
  to use rtransform at input ... or could go direct to transform ?
  Little point changing this, as will not help much 
  the issue is not so simple... 

* need to bring over the gtransforms (not the input transforms)
  (ie all distinct products of parent transforms in the tree) 
  ... hmm this will mean will need to collect all distinct 
  gtransforms off the node tree (TODO: digest for glm::mat4)



FIXED : Rotated geometry shows invalid boolean surfaces
----------------------------------------------------------

* without transform the boolean difference geometry 
  is rock solid, looking like real object from all angles

* with translation alone this ray trace still looks ok 

* with rotation get some crazy surfaces, looking like 
  bits of sphere which should have been boolean subtracted 
  from certain angles

* boolean machinery works by comparison of t values ... 
  so if different basis solids have different transforms
  ... but the issue doesnt look to be of interference between 
  shapes of different transforms

* propagation photons are seeing the invalid bits of sphere too

* tried increasing derived bbox to definitely contain the 
  geometry to see if an issue with bbox... but seems no difference

* rotating by 360 deg about z axis shows no issue, 
  so likely is caused by invalid axis-aligned assumption for box normals, 
  not a problem with transformation or bbox machinery  


MAYBE:

* box normal calc is assuming axis aligned, which is no longer true when 
  rotated in general ... try rotate by 90 degrees : this 
  makes the issue worse, rotating by 360 : no issue 

FIX

* transforming the normals from both box and sphere with the tr 
  looks to have fixed the issue




FIXED : All nodes in CSG tree with gtransformIdx  1  ?
--------------------------------------------------------

::

    ##bounds primIdx  0 partOffset  0 numParts  1 height  0 numNodes  1 tranBuffer_size   2 
    ##bounds primIdx  1 partOffset  1 numParts  7 height  2 numNodes  7 tranBuffer_size   2 
    ##hemi-pmt.cu:bounds primIdx 0 is_csg:0 min -1000.0000 -1000.0000 -1000.0000 max  1000.0000  1000.0000  1000.0000 
    ## bounds nodeIdx  4 depth  2 elev  0 partType  6 gtransformIdx  1 
    ## bounds nodeIdx  5 depth  2 elev  0 partType  5 gtransformIdx  1 
    ## bounds nodeIdx  2 depth  1 elev  1 partType  3 gtransformIdx  1 
    ## bounds nodeIdx  6 depth  2 elev  0 partType  6 gtransformIdx  1 
    ## bounds nodeIdx  7 depth  2 elev  0 partType  5 gtransformIdx  1 
    ## bounds nodeIdx  3 depth  1 elev  1 partType  3 gtransformIdx  1 
    ## bounds nodeIdx  1 depth  0 elev  2 partType  1 gtransformIdx  1 
    ##hemi-pmt.cu:bounds primIdx 1 is_csg:1 min  -325.4228  -355.3086  -185.1945 max   374.8348   486.3704   604.7207 


Fixed by not writing the bbox and getting nsphere::part to use nnode::part and then specialize,
but now get bad bbox for container which has disappeared in raytrace::

    ##bounds primIdx  0 partOffset  0 numParts  1 height  0 numNodes  1 tranBuffer_size   2 
    ##bounds primIdx  1 partOffset  1 numParts  7 height  2 numNodes  7 tranBuffer_size   2 
    ##hemi-pmt.cu:bounds primIdx 0 is_csg:0 min     0.0000     0.0000     0.0000 max     0.0000     0.0000     0.0000 
    ## bounds nodeIdx  4 depth  2 elev  0 partType  6 gtransformIdx  0 
    ## bounds nodeIdx  5 depth  2 elev  0 partType  5 gtransformIdx  0 
    ## bounds nodeIdx  2 depth  1 elev  1 partType  3 gtransformIdx  0 
    ## bounds nodeIdx  6 depth  2 elev  0 partType  6 gtransformIdx  1 
    ## bounds nodeIdx  7 depth  2 elev  0 partType  5 gtransformIdx  1 
    ## bounds nodeIdx  3 depth  1 elev  1 partType  3 gtransformIdx  0 
    ## bounds nodeIdx  1 depth  0 elev  2 partType  1 gtransformIdx  0 
    ##hemi-pmt.cu:bounds primIdx 1 is_csg:1 min  -273.6589  -355.3086  -300.0000 max   374.8348   300.0000   604.7207 



Dumping the GParts from OGeo shows that still have the bboxen and 
my gtransformIdx is being overwritten with a nodeIdx.

::


    In [11]: pt = np.load("/tmp/blyth/opticks/OGeo_makeAnalyticGeometry/analytic/partBuffer.npy")

    In [12]: pt
    Out[12]: 
    array([[[    0.    ,     0.    ,     0.    ,  1000.    ],
            [    0.    ,     0.    ,     0.    ,     0.    ],
            [-1000.    , -1000.    , -1000.    ,     0.    ],
            [ 1000.    ,  1000.    ,  1000.    ,     0.    ]],

           [[    0.    ,     0.    ,     0.    ,     0.    ],
            [    0.    ,     0.    ,     0.    ,     0.    ],
            [ -273.6589,  -355.3086,  -300.    ,     0.    ],
            [  374.8348,   300.    ,   604.7207,     0.    ]],

           [[    0.    ,     0.    ,     0.    ,     0.    ],
            [    0.    ,     0.    ,     0.    ,     0.    ],
            [ -100.    ,  -100.    ,  -300.    ,     0.    ],
            [  300.    ,   300.    ,   100.    ,     0.    ]],

           [[    0.    ,     0.    ,     0.    ,     0.    ],
            [    0.    ,     0.    ,     0.    ,     0.    ],
            [ -273.6589,  -355.3086,   -43.7731,     0.    ],
            [  374.8348,   293.1852,   604.7207,     0.    ]],

           [[  100.    ,   100.    ,  -100.    ,   150.1111],
            [    0.    ,     0.    ,     0.    ,     0.    ],
            [  -50.1111,   -50.1111,  -250.1111,     0.    ],
            [  250.1111,   250.1111,    50.1111,     0.    ]],

           [[  100.    ,   100.    ,  -100.    ,   200.    ],
            [    0.    ,     0.    ,     0.    ,     0.    ],
            [ -200.    ,  -200.    ,  -300.    ,     0.    ],
            [  200.    ,   200.    ,   100.    ,     0.    ]],

           [[    0.    ,     0.    ,   100.    ,   150.1111],
            [    0.    ,     0.    ,     0.    ,     0.    ],
            [ -192.7773,  -274.427 ,    37.1086,     0.    ],
            [  293.9532,   212.3035,   523.839 ,     0.    ]],

           [[    0.    ,     0.    ,   100.    ,   200.    ],
            [    0.    ,     0.    ,     0.    ,     0.    ],
            [ -200.    ,  -200.    ,  -100.    ,     0.    ],
            [  200.    ,   200.    ,   300.    ,     0.    ]]], dtype=float32)

::

    In [14]: pt = np.load("/tmp/blyth/opticks/OGeo_makeAnalyticGeometry/analytic/partBuffer.npy")

    In [15]: pt
    Out[15]: 
    array([[[    0.    ,     0.    ,     0.    ,  1000.    ],
            [    0.    ,     0.    ,     0.    ,     0.    ],
            [    0.    ,     0.    ,     0.    ,     0.    ],
            [    0.    ,     0.    ,     0.    ,     0.    ]],

           [[    0.    ,     0.    ,     0.    ,     0.    ],
            [    0.    ,     0.    ,     0.    ,     0.    ],
            [    0.    ,     0.    ,     0.    ,     0.    ],
            [    0.    ,     0.    ,     0.    ,     0.    ]],

           [[    0.    ,     0.    ,     0.    ,     0.    ],
            [    0.    ,     0.    ,     0.    ,     0.    ],
            [    0.    ,     0.    ,     0.    ,     0.    ],
            [    0.    ,     0.    ,     0.    ,     0.    ]],

           [[    0.    ,     0.    ,     0.    ,     0.    ],
            [    0.    ,     0.    ,     0.    ,     0.    ],
            [    0.    ,     0.    ,     0.    ,     0.    ],
            [    0.    ,     0.    ,     0.    ,     0.    ]],

           [[  100.    ,   100.    ,  -100.    ,   150.1111],
            [    0.    ,     0.    ,     0.    ,     0.    ],
            [    0.    ,     0.    ,     0.    ,     0.    ],
            [    0.    ,     0.    ,     0.    ,     0.    ]],

           [[  100.    ,   100.    ,  -100.    ,   200.    ],
            [    0.    ,     0.    ,     0.    ,     0.    ],
            [ -200.    ,  -200.    ,  -300.    ,     0.    ],
            [  200.    ,   200.    ,   100.    ,     0.    ]],

           [[    0.    ,     0.    ,   100.    ,   150.1111],
            [    0.    ,     0.    ,     0.    ,     0.    ],
            [    0.    ,     0.    ,     0.    ,     0.    ],
            [    0.    ,     0.    ,     0.    ,     0.    ]],

           [[    0.    ,     0.    ,   100.    ,   200.    ],
            [    0.    ,     0.    ,     0.    ,     0.    ],
            [ -200.    ,  -200.    ,  -100.    ,     0.    ],
            [  200.    ,   200.    ,   300.    ,     0.    ]]], dtype=float32)





    In [13]: pt.view(np.uint32)
    Out[13]: 
    array([[[         0,          0,          0, 1148846080],
            [         0,          0,        123,          0],
            [3296329728, 3296329728, 3296329728,          6],
            [1148846080, 1148846080, 1148846080,          0]],

           [[         0,          0,          0,          0],
            [         0,          1,        124,          0],
            [3280524376, 3283199872, 3281387520,          1],
            [1136356060, 1133903872, 1142369824,          1]],

           [[         0,          0,          0,          0],
            [         0,          2,        124,          0],
            [3267887104, 3267887104, 3281387520,          3],
            [1133903872, 1133903872, 1120403456,          1]],

           [[         0,          0,          0,          0],
            [         0,          3,        124,          0],
            [3280524376, 3283199872, 3257866152,          3],
            [1136356060, 1133680564, 1142369824,          1]],

           [[1120403456, 1120403456, 3267887104, 1125522543],
            [         0,          4,        124,          0],
            [3259527612, 3259527612, 3279559791,          6],
            [1132076143, 1132076143, 1112043964,          1]],

           [[1120403456, 1120403456, 3267887104, 1128792064],
            [         0,          5,        124,          0],
            [3276275712, 3276275712, 3281387520,          5],
            [1128792064, 1128792064, 1120403456,          1]],

           [[         0,          0, 1120403456, 1125522543],
            [         0,          6,        124,          0],
            [3275802366, 3280549543, 1108635432,          6],
            [1133705730, 1129598387, 1141044658,          1]],

           [[         0,          0, 1120403456, 1128792064],
            [         0,          7,        124,          0],
            [3276275712, 3276275712, 3267887104,          5],
            [1128792064, 1128792064, 1133903872,          1]]], dtype=uint32)




input csg very spartan
-----------------------

* but gets imported by NCSG into nnode treem and then exported 



::

    In [4]: n = np.load("/tmp/blyth/opticks/tboolean-csg-two-box-minus-sphere-interlocked-py-/1/nodes.npy")

    In [5]: n
    Out[5]: 
    array([[[   0.    ,    0.    ,    0.    ,    0.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]],

           [[   0.    ,    0.    ,    0.    ,    0.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]],

           [[   0.    ,    0.    ,    0.    ,    0.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]],

           [[ 100.    ,  100.    , -100.    ,  150.1111],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]],

           [[ 100.    ,  100.    , -100.    ,  200.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]],

           [[   0.    ,    0.    ,  100.    ,  150.1111],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]],

           [[   0.    ,    0.    ,  100.    ,  200.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]]], dtype=float32)


    In [6]: n.view(np.int32)
    Out[6]: 
    array([[[          0,           0,           0,           0],
            [          0,           0,           0,           0],
            [          0,           0,           0,           1],      CSG_UNION 
            [          0,           0,           0,           1]],    <----- whats this 1 ? IT IS THE RTRANSFORM REFERENCE

           [[          0,           0,           0,           0],
            [          0,           0,           0,           0],
            [          0,           0,           0,           3],      CSG_DIFFERENCE
            [          0,           0,           0,           0]],

           [[          0,           0,           0,           0],
            [          0,           0,           0,           0],
            [          0,           0,           0,           3],       CSG_DIFFERENCE
            [          0,           0,           0,           0]],

           [[ 1120403456,  1120403456, -1027080192,  1125522543],
            [          0,           0,           0,           0],
            [          0,           0,           0,           6],      CSG_BOX
            [          0,           0,           0,           0]],

           [[ 1120403456,  1120403456, -1027080192,  1128792064],
            [          0,           0,           0,           0],
            [          0,           0,           0,           5],      CSG_SPHERE
            [          0,           0,           0,           0]],

           [[          0,           0,  1120403456,  1125522543],
            [          0,           0,           0,           0],
            [          0,           0,           0,           6],       CSG_BOX
            [          0,           0,           0,           0]],

           [[          0,           0,  1120403456,  1128792064],
            [          0,           0,           0,           0],
            [          0,           0,           0,           5],       CSG_SPHERE
            [          0,           0,           0,           0]]], dtype=int32)



    simon:opticks blyth$ sysrap-csg

    typedef enum {
        CSG_ZERO=0,
        CSG_UNION=1,
        CSG_INTERSECTION=2,
        CSG_DIFFERENCE=3,
        CSG_PARTLIST=4,   

        CSG_SPHERE=5,
           CSG_BOX=6,
       CSG_ZSPHERE=7,
         CSG_ZLENS=8,
           CSG_PMT=9,
         CSG_PRISM=10,
          CSG_TUBS=11,
     CSG_UNDEFINED=12

    } OpticksCSG_t ; 
       






can partlist work with derived bbox ? does not look like it
---------------------------------------------------------------

* suspect not, contrary to recollection it aint just z that is setup...
* this means need to work with different layouts for CSG and PARTLIST 

  * where to effect the split...  





::

    In [1]: p = np.load("/usr/local/opticks/opticksdata/export/DayaBay/GPmt/1/GPmt.npy")

    In [2]: p
    Out[2]: 
    array([[[   0.    ,    0.    ,   69.    ,  102.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [-101.1682, -101.1682,  -23.8382,    0.    ],
            [ 101.1682,  101.1682,   56.    ,    0.    ]],

           [[   0.    ,    0.    ,   43.    ,  102.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [-101.1682, -101.1682,   56.    ,    0.    ],
            [ 101.1682,  101.1682,  100.0698,    0.    ]],

           [[   0.    ,    0.    ,    0.    ,  131.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [ -84.5402,  -84.5402,  100.0698,    0.    ],
            [  84.5402,   84.5402,  131.    ,    0.    ]],

           [[   0.    ,    0.    ,  -84.5   ,   42.25  ],
            [ 169.    ,    0.    ,    0.    ,    0.    ],
            [ -42.25  ,  -42.25  , -169.    ,    0.    ],
            [  42.25  ,   42.25  ,  -23.8382,    0.    ]],

           [[   0.    ,    0.    ,   69.    ,   99.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [ -98.1428,  -98.1428,  -21.8869,    0.    ],
            [  98.1428,   98.1428,   56.    ,    0.    ]],

           [[   0.    ,    0.    ,   43.    ,   99.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [ -98.1428,  -98.1428,   56.    ,    0.    ],
            [  98.1428,   98.1428,   98.0465,    0.    ]],

           [[   0.    ,    0.    ,    0.    ,  128.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [ -82.2854,  -82.2854,   98.0465,    0.    ],
            [  82.2854,   82.2854,  128.    ,    0.    ]],

           [[   0.    ,    0.    ,  -81.5   ,   39.25  ],
            [ 166.    ,    0.    ,    0.    ,    0.    ],
            [ -39.25  ,  -39.25  , -164.5   ,    0.    ],
            [  39.25  ,   39.25  ,  -21.8869,    0.    ]],

           [[   0.    ,    0.    ,    0.    ,  127.95  ],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [ -82.2478,  -82.2478,   98.0128,    0.    ],
            [  82.2478,   82.2478,  127.95  ,    0.    ]],

           [[   0.    ,    0.    ,   43.    ,   98.95  ],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [ -98.0932,  -98.0932,   55.9934,    0.    ],
            [  98.0932,   98.0932,   98.0128,    0.    ]],

           [[   0.    ,    0.    ,   69.    ,   98.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [ -97.1514,  -97.1514,  -29.    ,    0.    ],
            [  97.1514,   97.1514,   56.1313,    0.    ]],

           [[   0.    ,    0.    ,  -81.5   ,   27.5   ],
            [ 166.    ,    0.    ,    0.    ,    0.    ],
            [ -27.5   ,  -27.5   , -164.5   ,    0.    ],
            [  27.5   ,   27.5   ,    1.5   ,    0.    ]]], dtype=float32)

    In [3]: p.view(np.int32)
    Out[3]: 
    array([[[          0,           0,  1116340224,  1120665600],
            [          0,           1,           0,           0],
            [-1026927077, -1026927077, -1044466509,           5],
            [ 1120556571,  1120556571,  1113587712,           0]],

           [[          0,           0,  1110179840,  1120665600],
            [          0,           2,           0,           0],
            [-1026927077, -1026927077,  1113587712,           5],
            [ 1120556571,  1120556571,  1120412601,           0]],

           [[          0,           0,           0,  1124270080],
            [          0,           3,           0,           0],
            [-1029106542, -1029106542,  1120412601,           5],
            [ 1118377106,  1118377106,  1124270080,           0]],

           [[          0,           0, -1029111808,  1109983232],
            [ 1126760448,           4,           0,           1],
            [-1037500416, -1037500416, -1020723200,          11],
            [ 1109983232,  1109983232, -1044466509,           0]],

           [[          0,           0,  1116340224,  1120272384],
            [          0,           5,           0,           0],
            [-1027323625, -1027323625, -1045489543,           5],
            [ 1120160023,  1120160023,  1113587712,           1]],

           [[          0,           0,  1110179840,  1120272384],
            [          0,           6,           0,           0],
            [-1027323625, -1027323625,  1113587712,           5],
            [ 1120160023,  1120160023,  1120147408,           1]],

           [[          0,           0,           0,  1124073472],
            [          0,           7,           0,           0],
            [-1029402084, -1029402084,  1120147408,           5],
            [ 1118081564,  1118081564,  1124073472,           1]],

           [[          0,           0, -1029505024,  1109196800],
            [ 1126563840,           8,           0,           1],
            [-1038286848, -1038286848, -1021018112,          11],
            [ 1109196800,  1109196800, -1045489543,           1]],

           [[          0,           0,           0,  1124066918],
            [          0,           9,           0,           0],
            [-1029407013, -1029407013,  1120142989,           5],
            [ 1118076635,  1118076635,  1124066918,           2]],

           [[          0,           0,  1110179840,  1120265830],
            [          0,          10,           0,           0],
            [-1027330122, -1027330122,  1113585991,           5],
            [ 1120153526,  1120153526,  1120142989,           2]],

           [[          0,           0,  1116340224,  1120141312],
            [          0,          11,           0,           0],
            [-1027453562, -1027453562, -1041760256,           5],
            [ 1120030086,  1120030086,  1113622135,           3]],

           [[          0,           0, -1029505024,  1104936960],
            [ 1126563840,          12,           0,           0],
            [-1042546688, -1042546688, -1021018112,          11],
            [ 1104936960,  1104936960,  1069547520,           4]]], dtype=int32)

    In [4]: 



move bbox calc to GPU
-----------------------

::

    ##test_tranBuffer tr
       0.805    0.506   -0.311    0.000
      -0.311    0.805    0.506    0.000
       0.506   -0.311    0.805    0.000
       0.000    0.000  200.000    1.000
    tr0
       0.805    0.506   -0.311    0.000
    tr1
      -0.311    0.805    0.506    0.000
    tr2
       0.506   -0.311    0.805    0.000
    tr3
       0.000    0.000  200.000    1.000

    ##test_tranBuffer irit
       0.805   -0.311    0.506    0.000
       0.506    0.805   -0.311    0.000
      -0.311    0.506    0.805    0.000
      62.123 -101.176 -160.948    1.000

    ##test_transform_bbox tr
       0.805    0.506   -0.311    0.000
      -0.311    0.805    0.506    0.000
       0.506   -0.311    0.805    0.000
       0.000    0.000  200.000    1.000

    ##test_transform_bbox min -162.123 -162.123   37.877 max  162.123  162.123  362.123 



    elta:optixu blyth$ NBBoxTest

    (  0)       0.805       0.506      -0.311       0.000 
    (  0)      -0.311       0.805       0.506       0.000 
    (  0)       0.506      -0.311       0.805       0.000 
    (  0)       0.000       0.000     200.000       1.000 
            tr  0.805   0.506  -0.311   0.000 
               -0.311   0.805   0.506   0.000 
                0.506  -0.311   0.805   0.000 
                0.000   0.000 200.000   1.000 

         tr[0]  0.805   0.506  -0.311   0.000 

         tr[1] -0.311   0.805   0.506   0.000 

         tr[2]  0.506  -0.311   0.805   0.000 

         tr[3]  0.000   0.000 200.000   1.000 

    bb  mi  (-100.00 -100.00 -100.00)  mx  ( 100.00  100.00  100.00)  
    tbb  mi  (-162.12 -162.12   37.88)  mx  ( 162.12  162.12  362.12)  





SDF
------

* Where to hold the transform in nnode trees and CSG trees ?

 * G4 allows the RHS of a boolean combination to be transformed using 
   a transform that lives with the combination



* use glm::mat4 ?


local/global transforms ?
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    111 double nunion::operator()(double px, double py, double pz)
    112 {

    ///    just transform px,py,pz here only  ?

    113     assert( left && right );
    114     double l = (*left)(px, py, pz) ;
    115     double r = (*right)(px, py, pz) ;
    116     return fmin(l, r);
    117 }


Perhaps can just locally apply the transform ? to the coordinates
passed down the tree ? Relying on subsequent transforms transforming 
again the transformed coordinates... this would be simplest.

The alternative would be to traverse up the tree thru parent 
links collecting and multiplying transforms and store that 
as a global transfrom within each node to apply to global coordinates.

Actually its not clear how to use global transforms as the evaluation is done
treewise ... with each node not knowing where it is in the tree ?

BUT: for internal nodes the coordinates are not actually used, they are 
just being passed down the tree until reach the leaves/primitives ... so this 
means can collect ancestor transforms into the primitives : this is 
what will need to do on GPU, so actually its better to take same approach on CPU 


* adopted globaltransform held in primitive, which is obtained at deserialization (in NCSG)
  from product of ancestor node transforms


Transform references
----------------------

::

     09 // only used for CSG operator nodes
     10 enum {
     11     RTRANSFORM_J = 3,
     12     RTRANSFORM_K = 3
     13 };   // q3.u.w
     14 

     58 enum {
     59     NODEINDEX_J = 3,
     60     NODEINDEX_K = 3
     61 };  // q3.u.w 


* input serialization has rtransform references in CSG operator nodes
* these are set on the appropriate primitive nnode in the in memory model ...
* BUT what about on GPU, want to avoid tree chasing BUT 


Need to make space in part/node buffer for transform referencing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* for CSG with transforms the old fixed bb.min, bb.max 
  no longer cuts it ... actually it could do, just means the 
  transforming the bbox is done CPU side 

* the critical thing is that the bbox occupies 6*32bits 
  out of the total 16*32 bits ... i think the reasoning behind this
  was for z-range selection in the partlist approach 

* can adopt different layout in CSG mode

* bbox calc only done once in bounds code, so it has no performance cost 


Transforming Rays
-------------------

The below needs to pass a reference to the ray to the intersects
and the transform can happen here.

::

    float3:  ray.direction, ray.origin 

::

    128 static __device__
    129 void intersect_part(unsigned partIdx, const float& tt_min, float4& tt  )
    130 {
    131     quad q0, q2 ;
    132     q0.f = partBuffer[4*partIdx+0];
    133     q2.f = partBuffer[4*partIdx+2];
    134 
    135     OpticksCSG_t csgFlag = (OpticksCSG_t)q2.u.w ;
    136 
    137     //if(partIdx > 1)
    138     //rtPrintf("[%5d] intersect_part partIdx %u  csgFlag %u \n", launch_index.x, partIdx, csgFlag );
    139 
    140     switch(csgFlag)
    141     {
    142         case CSG_SPHERE: intersect_sphere(q0,tt_min, tt )  ; break ;
    143         case CSG_BOX:    intersect_box(   q0,tt_min, tt )  ; break ;
    144     }
    145 }




Transforms GPU side 
--------------------

* does GPU need *tr* OR perhaps only *irit* will do, as primary action 
  is transforming impinging rays not directly geometry 

* transforming bbox with need the *tr*, transforming rays will need the *irit*

* optix Matrix4x4 uses row-major, Opticks standard follows OpenGL : column-major

::

    9.005 Are OpenGL matrices column-major or row-major?

    For programming purposes, OpenGL matrices are 16-value arrays with base vectors
    laid out contiguously in memory. The translation components occupy the 13th,
    14th, and 15th elements of the 16-element matrix, where indices are numbered
    from 1 to 16 as described in section 2.11.2 of the OpenGL 2.1 Specification.

    Column-major versus row-major is purely a notational convention. Note that
    post-multiplying with column-major matrices produces the same result as
    pre-multiplying with row-major matrices. The OpenGL Specification and the
    OpenGL Reference Manual both use column-major notation. You can use any
    notation, as long as it's clearly stated.


::

    /Developer/OptiX/include/optixu/optixu_matrix_namespace.h

    100   template <unsigned int M, unsigned int N>
    101   class Matrix
    102   {
    103   public:
    ...
    169   private:
    170       /** The data array is stored in row-major order */
    171       float m_data[M*N];
    172   };
    173 
       
    421   // Multiply matrix4x4 by float4
    422   OPTIXU_INLINE RT_HOSTDEVICE float4 operator*(const Matrix<4,4>& m, const float4& vec )
    423   {
    424     float4 temp;
    425     temp.x  = m[ 0] * vec.x +
    426               m[ 1] * vec.y +
    427               m[ 2] * vec.z +
    428               m[ 3] * vec.w;
    429     temp.y  = m[ 4] * vec.x +
    430               m[ 5] * vec.y +
    431               m[ 6] * vec.z +
    432               m[ 7] * vec.w;
    433     temp.z  = m[ 8] * vec.x +
    434               m[ 9] * vec.y +
    435               m[10] * vec.z +
    436               m[11] * vec.w;
    437     temp.w  = m[12] * vec.x +
    438               m[13] * vec.y +
    439               m[14] * vec.z +
    440               m[15] * vec.w;
    441 
    442     return temp;
    443   }


    709   typedef Matrix<2, 2> Matrix2x2;
    710   typedef Matrix<2, 3> Matrix2x3;
    711   typedef Matrix<2, 4> Matrix2x4;
    712   typedef Matrix<3, 2> Matrix3x2;
    713   typedef Matrix<3, 3> Matrix3x3;
    714   typedef Matrix<3, 4> Matrix3x4;
    715   typedef Matrix<4, 2> Matrix4x2;
    716   typedef Matrix<4, 3> Matrix4x3;
    717   typedef Matrix<4, 4> Matrix4x4;
    718 




Transforming BBox ?
---------------------

* http://dev.theomader.com/transform-bounding-boxes/
* http://www.cs.unc.edu/~zhangh/technotes/bbox.pdf

* https://www.geometrictools.com/Documentation/AABBForTransformedAABB.pdf
* https://github.com/erich666/GraphicsGems/blob/master/gems/TransBox.c
* http://www.akshayloke.com/2012/10/22/optimized-transformations-for-aabbs/



Models
-------

* input python model opticks.dev.csg.csg.CSG
* numpy array serialization
* NCSG created nnode model  


Where to hang the transform ?
--------------------------------

parent.rtransform OR node.transform ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* transform reference on CSG operation node is advantageous, as no space pressure there

  * actually above "advantage" is conflating the serialization with the in memory nnode model, 
    the in nnode model does not have any space issues, and it does not need to 
    precisely follow what the serialization does

* so can define and serialize using rtransform and then deserialize onto transforms 
  directly on nodes as that is easier in usage 

* not so clear that node.transform is easier in usage... as 
  would mean that every primitive needs to implement coordinate transformations 
  handling as opposed to just the 3 CSG operation nodes



