Triangulation of CSG defined geometry using SDF
==================================================


Dual Contouring Sample
-------------------------

* works, but very slow : time dominated by ConstructOctreeNodes
* a few small changes halves the time ... but still slow


::

    2017-03-24 12:21:36.557 INFO  [1542447] [*GParts::make@122] GParts::make NCSG  path /tmp/blyth/opticks/tboolean-csg-two-box-minus-sphere-interlocked-py-/0.npy sh 1,4,4 spec Rock//perfectAbsorbSurface/Vacuum type box
    2017-03-24 12:21:36.557 INFO  [1542447] [*NDualContouringSample::operator@75] NDualContouringSample  xyzExtent 303 ijkExtent 64 bbce  (  50.00   50.00    0.00  300.00)  ce 50.0000,50.0000,0.0000,4.7344 ilow -64,-64,-64
    CheckDomain  size 128 min (-64,-64,-64) ce  (50,50,0,4.73438)
    corner  0 ijk (-64,-64,-64) xyz (      -253,      -253,      -303) -->    202.889
    corner  1 ijk (-64,-64, 64) xyz (      -253,      -253,       303) -->    102.889
    corner  2 ijk (-64, 64,-64) xyz (      -253,       353,      -303) -->    202.889
    corner  3 ijk (-64, 64, 64) xyz (      -253,       353,       303) -->    202.889
    corner  4 ijk ( 64,-64,-64) xyz (       353,      -253,      -303) -->    202.889
    corner  5 ijk ( 64,-64, 64) xyz (       353,      -253,       303) -->    202.889
    corner  6 ijk ( 64, 64,-64) xyz (       353,       353,      -303) -->    102.889
    corner  7 ijk ( 64, 64, 64) xyz (       353,       353,       303) -->    202.889
    2017-03-24 12:21:44.194 INFO  [1542447] [*NDualContouringSample::operator@115]  vertices 1810
    2017-03-24 12:21:44.194 INFO  [1542447] [*NDualContouringSample::operator@116]  indices  12270
    2017-03-24 12:21:44.196 INFO  [1542447] [NDualContouringSample::report@53] NDualContouringSample::
    2017-03-24 12:21:44.197 INFO  [1542447] [NDualContouringSample::report@54] NDualContouringSample log2size 7 octreeSize 128 threshold 1 scale_bb 1.01 ilow -64,-64,-64
    2017-03-24 12:21:44.197 INFO  [1542447] [TimesTable::dump@103] TimesTable::dump filter: NONE
              0.000      t_absolute        t_delta
              0.000           0.000          0.000 : _BuildOctree
              0.000           0.000          0.000 : _ConstructOctreeNodes
              7.615           7.616          7.615 : ConstructOctreeNodes
              0.000           7.616          0.000 : _SimplifyOctree
              0.020           7.636          0.020 : SimplifyOctree
              0.000           7.636          0.000 : BuildOctree
              0.000           7.636          0.000 : _GenerateMeshFromOctree
              0.001           7.637          0.001 : GenerateMeshFromOctree
              0.000           7.637          0.000 : _CollectTriangles
              0.002           7.639          0.002 : CollectTriangles
    2017-03-24 12:21:44.197 INFO  [1542447] [*GMaker::makeFromCSG@130] GMaker::makeFromCSG tessa DCS numTris 4090  mi  (-150.03 -150.03 -250.33)  mx  ( 250.03  250.03  250.33)  
    2017-03-24 12:21:44.198 INFO  [1542447] [GMesh::updateBounds@1283] GMesh::updateBounds mesh with verts,  g_instance_count 6



Avoid passing function by value, instead just pass pointer to function thru all those levels of recursion : nearly halves the time::

    2017-03-24 12:44:40.951 INFO  [1548682] [NDualContouringSample::report@54] NDualContouringSample log2size 7 octreeSize 128 threshold 1 scale_bb 1.01 ilow -64,-64,-64
    2017-03-24 12:44:40.951 INFO  [1548682] [TimesTable::dump@103] TimesTable::dump filter: NONE
              0.000      t_absolute        t_delta
              0.000           0.000          0.000 : _BuildOctree
              0.000           0.000          0.000 : _ConstructOctreeNodes
              4.175           4.175          4.175 : ConstructOctreeNodes
              0.000           4.175          0.000 : _SimplifyOctree
              0.019           4.194          0.019 : SimplifyOctree
              0.000           4.194          0.000 : BuildOctree
              0.000           4.194          0.000 : _GenerateMeshFromOctree
              0.001           4.195          0.001 : GenerateMeshFromOctree
              0.000           4.195          0.000 : _CollectTriangles
              0.002           4.197          0.002 : CollectTriangles
    2017-03-24 12:44:40.951 INFO  [1548682] [*GMaker::makeFromCSG@130] GMaker::makeFromCSG tessa DCS numTris 4090  mi  (-150.03 -150.03 -250.33)  mx  ( 250.03  250.03  250.33)  




Special case leaf creation with lookahead one level corner check, doesnt improve much::

    2017-03-24 13:48:06.759 INFO  [1568949] [NDualContouringSample::report@54] NDualContouringSample log2size 7 octreeSize 128 threshold 1 scale_bb 1.01 ilow -64,-64,-64
    2017-03-24 13:48:06.759 INFO  [1568949] [TimesTable::dump@103] TimesTable::dump filter: NONE
              0.000      t_absolute        t_delta
              0.000           0.000          0.000 : _BuildOctree
              0.000           0.000          0.000 : _ConstructOctreeNodes
              4.002           4.003          4.002 : ConstructOctreeNodes
              0.000           4.003          0.000 : _SimplifyOctree
              0.020           4.023          0.020 : SimplifyOctree
              0.000           4.023          0.000 : BuildOctree
              0.000           4.023          0.000 : _GenerateMeshFromOctree
              0.001           4.024          0.001 : GenerateMeshFromOctree
              0.000           4.025          0.000 : _CollectTriangles
              0.002           4.027          0.002 : CollectTriangles
    2017-03-24 13:48:06.759 INFO  [1568949] [*GMaker::makeFromCSG@130] GMaker::makeFromCSG tessa DCS numTris 4090  mi  (-150.03 -150.03 -250.33)  mx  ( 250.03  250.03  250.33)  
    2017-03-24 13:48:06.760 INFO  [1568949] [GMesh::updateBounds@1283] GMesh::updateBounds mesh with verts,  g_instance_count 6


Recursive HasChildren lookahead adds time::

    2017-03-24 14:24:03.515 INFO  [1578213] [TimesTable::dump@103] TimesTable::dump filter: NONE
              0.000      t_absolute        t_delta
              0.000           0.000          0.000 : _BuildOctree
              0.000           0.000          0.000 : _ConstructOctreeNodes
              5.603           5.604          5.603 : ConstructOctreeNodes
              0.000           5.604          0.000 : _SimplifyOctree
              0.021           5.624          0.021 : SimplifyOctree
              0.000           5.624          0.000 : BuildOctree
              0.000           5.624          0.000 : _GenerateMeshFromOctree
              0.001           5.626          0.001 : GenerateMeshFromOctree
              0.000           5.626          0.000 : _CollectTriangles
              0.002           5.628          0.002 : CollectTriangles
    2017-03-24 14:24:03.515 INFO  [1578213] [*GMaker::makeFromCSG@130] GMaker::makeFromCSG tessa DCS numTris 4090  mi  (-150.03 -150.03 -250.33)  mx  ( 250.03  250.03  250.33)  


::

    ConstructOctreeNodes count 299593


    In [101]: print "\n".join(map(oct_, range(16)))
                       0                    1                    1                    1
                       1                    2                    8                    9
                       2                    4                   64                   73
                       3                    8                  512                  585
                       4                   16                4,096                4,681
                       5                   32               32,768               37,449
                       6                   64              262,144              299,593  <<<
                       7                  128            2,097,152            2,396,745
                       8                  256           16,777,216           19,173,961
                       9                  512          134,217,728          153,391,689
                      10                1,024        1,073,741,824        1,227,133,513
                      ------------------------------------------------------------------
                      11                2,048        8,589,934,592        9,817,068,105
                      12                4,096       68,719,476,736       78,536,544,841
                      13                8,192      549,755,813,888      628,292,358,729
                      14               16,384    4,398,046,511,104    5,026,338,869,833
                      15               32,768   35,184,372,088,832   40,210,710,958,665

