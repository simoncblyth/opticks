#!/usr/bin/env python
"""
Genstep Sequence Material Mismatch
===================================

Comparing material codes of the first intersection with the 
material code of the genstep observe many mismatches.

Current understanding of problem
------------------------------------

::

     GdLS     |Ac|   LS    |Ac|   MO
              |  |         |  |      
              |  |         |  |
       *---------x         |  |
              |  |         |  |
              |  |         |  | 
              |  |         |  |
       *------x  |         |  |
              |  |         |  |

Many (~14%) photons with gensteps in Gd have a 1st boundary intersection 
with Ac/LS when expected to intersect with Gd/Ac

Questions
-----------

* geometric preponderances ? particular angles or closeness ? particular pieces of geometry : lids ?
* geometric epsilon 
* thin acrylic problem ?

  * make simplified geometry: concentric spheres (or boxes) with varying Ac thickness 


Material Codes
-----------------

*gs.MaterialIndex*
      from Geant4, converted into wavelength texture line number in G4StepNPY

*m1/m2/su*
      OptiX intersects with a triangle, reading the assigned boundary code, sign 
      from cosTheta of angle between photon and triangle normal
      pulled from optical buffer using the absolute boundary code and its sign
      to map inner/outer to m1/m2 


OptiX epsilon
--------------

::

     324 void OEngine::preprocess()
     325 {
     326     LOG(info)<< "OEngine::preprocess";
     327 
     328     m_context[ "scene_epsilon"]->setFloat(m_composition->getNear());
     329 
     330     float pe = 0.1f ;
     331     m_context[ "propagate_epsilon"]->setFloat(pe);  // TODO: check impact of changing propagate_epsilon
     332 

cu/generate.cu::

    171         rtTrace(top_object, optix::make_Ray(p.position, p.direction, propagate_ray_type, propagate_epsilon, RT_DEFAULT_MAX), prd );



Extent of Problem : 9 percent
-------------------------------

Test using Aux buffer containing in 1st slot of each set of 10  

* m1
* m2
* "m1" translated from genstep material line
* signed boundary code

::

    In [1]: a = auc_(1).reshape(-1,10,4)
    INFO:env.numerics.npy.types:loading /usr/local/env/dayabay/aucerenkov/1.npy 
    -rw-r--r--  1 blyth  staff  49027360 Sep 19 13:01 /usr/local/env/dayabay/aucerenkov/1.npy

    In [39]: aa = a[:500000]   ## exclude tail where there are geometry misses (still using sub-geometry), so no boundary-m1 to compare with 

    # TODO: move to fuller geometry include radslabs around pool to avoid misses, can still skip RPC 

    In [40]: aa
    Out[40]: 
    array([[[  6,   6,   6, -12],     ## 1st and 3rd indices here should match  boundary-m1 and genstep-m1 
            [  6,  12,   4, -13],
            [ 12,   4,   0, -14],
            ..., 
            [  0,   0,   0,   0],
            [  0,   0,   0,   0],
            [  0,   0,   0,   0]],


    In [47]: ix = np.arange(len(aa))[aa[:,0,0] != aa[:,0,2]]

    In [48]: ix
    Out[48]: array([  3006,   8521,   8524, ..., 497348, 497349, 497350])

    In [49]: len(ix)
    Out[49]: 44479

    In [50]: len(aa)
    Out[50]: 500000

    In [54]: float(len(ix))/float(len(aa))   ## 9 percent level mismatch 
    Out[54]: 0.088958


How often can match be made using material from other side of boundary ? Only 1 percent
----------------------------------------------------------------------------------------

Suggests not a geometric normal (vertex winding order) problem. 

::

    In [58]: aa[:,0][ix]
    Out[58]: 
    array([[  4,  14,   6, -20],
           [  4,  11,   6, -23],
           [  4,  11,   6, -23],
           ..., 
           [  4,   3,   3, -25],
           [  4,  14,   3, -20],
           [  4,  14,   3, -20]], dtype=int16)

::

    In [64]: aaa = aa[:,0][ix]

    In [66]: farside = np.arange(len(aaa))[aaa[:,1] == aaa[:,2]]

    In [70]: float(len(farside))/len(aaa)
    Out[70]: 0.011398637559297646



Is there preponderance of particular genstep materials with mismatch ? yes: Gd dominates
-------------------------------------------------------------------------------------------

::

    In [81]: count_unique(aaa[:,2])
    Out[81]: 
    array([[    1, 43666],
           [    2,   419],
           [    3,   390],
           [    4,     1],
           [    6,     3]])

    In [82]: im_ = lambda _:im.get(_,'?')

    In [83]: map(im_,[1,2,3,4,6])
    Out[83]: ['GdDopedLS', 'LiquidScintillator', 'Acrylic', 'MineralOil', 'IwsWater']


How about the boundary ?  predominantly imat:Acrylic omat:LiquidScintillator
-------------------------------------------------------------------------------

TODO: visualize where the problem is, suspect AD lids 

::

    In [85]: count_unique(aaa[:,3])
    Out[85]: 
    array([[  -25,   125],
           [  -23,     5],
           [  -21,     1],
           [  -20,    12],
           [  -18,   123],
           [  -17,   137],
           [  -16,     8],
           [  -15,    11],
           [   15,   411],
           [   16,   103],
           [   17, 43543]])     imat:Acrylic omat:LiquidScintillator

::

    delta:ggeo blyth$ ./GBoundaryLibMetadata.py 
    INFO:__main__:['./GBoundaryLibMetadata.py']
      0 :  1 :                      Vacuum                    Vacuum                         -                         - 
      1 :  2 :                        Rock                    Vacuum                         -                         - 
      2 :  3 :                         Air                      Rock                         -                         - 
      3 :  4 :                         PPE                       Air                         - __dd__Geometry__PoolDetails__NearPoolSurfaces__NearPoolCoverSurface 
      4 :  5 :                   Aluminium                       Air                         -                         - 
      5 :  6 :                        Foam                 Aluminium                         -                         - 
      6 :  7 :                         Air                       Air                         -                         - 
      7 :  8 :                   DeadWater                      Rock                         -                         - 
      8 :  9 :                       Tyvek                 DeadWater                         - __dd__Geometry__PoolDetails__NearPoolSurfaces__NearDeadLinerSurface 
      9 : 10 :                    OwsWater                     Tyvek __dd__Geometry__PoolDetails__NearPoolSurfaces__NearOWSLinerSurface                         - 
     10 : 11 :                       Tyvek                  OwsWater                         -                         - 
     11 : 12 :                    IwsWater                  IwsWater                         -                         - 
     12 : 13 :              StainlessSteel                  IwsWater                         - __dd__Geometry__AdDetails__AdSurfacesNear__SSTWaterSurfaceNear1 
     13 : 14 :                  MineralOil            StainlessSteel __dd__Geometry__AdDetails__AdSurfacesAll__SSTOilSurface                         - 
     14 : 15 :                     Acrylic                MineralOil                         -                         - 
     15 : 16 :          LiquidScintillator                   Acrylic                         -                         - 
     16 :*17*:                     Acrylic        LiquidScintillator                         -                         - 
     17 : 18 :                   GdDopedLS                   Acrylic                         -                         - 
     18 : 19 :                   GdDopedLS        LiquidScintillator                         -                         - 
     19 : 20 :                       Pyrex                MineralOil                  



What boundaries do GdLs gensteps have ?
------------------------------------------

::

    In [96]: gd = np.arange(len(aa))[aa[:,0,2] == 1]

    In [97]: gd
    Out[97]: array([ 90402,  90403,  90404, ..., 453773, 453774, 453775])

    In [98]: aa[gd]
    Out[98]: 
    array([[[ 1,  3,  1, 18],
            [ 1,  3,  1, 18],
            [ 0,  0,  0,  0],
            ..., 
            [ 0,  0,  0,  0],
            [ 0,  0,  0,  0],
            [ 0,  0,  0,  0]],

    In [99]: aa[gd][:,0,3]
    Out[99]: array([18, 18, 18, ..., 17, 17, 17], dtype=int16)

    In [100]: count_unique(aa[gd][:,0,3])
    Out[100]: 
    array([[   -18,    123],
           [    17,  43543],
           [    18, 308763]])

::

    In [103]: 43543./308763.
    Out[103]: 0.14102402166062644

::

         +1

     16 :*17*:                     Acrylic        LiquidScintillator                         -                         - 
     17 :*18*:                   GdDopedLS                   Acrylic                         -                         - 
 


What 1st boundaries do LS gensteps have ?  Only 0.4% mismatch
---------------------------------------------------------------

Hmm low level mismatch ? Something special regards inner AV.

::

    In [105]: ls = np.arange(len(aa))[aa[:,0,2] == 2]

    In [125]: ac = np.arange(len(aa))[aa[:,0,2] == 3]

    In [106]: count_unique(aa[ls][:,0,3])
    Out[106]: 
    array([[  -17, 51348],   #  Ac/LS but negated so from Ls 
           [  -16,     8],   #  LS/Ac but negated so from Ac?  
           [   15,   411],   #  Ac/MO
           [   16, 43249]])  #  LS/Ac 


::

     GdLS     |Ac|       LS    |Ac|   MO
              |  |             |  |      
              |  | -17         |  |
              |  x------*      |  |
              |  |             |  |
              |  |             |  | 
              |  |        +16  |  |
              |  |      *------x  |
              |  |             |  |


::

         (+1)
     14 : 15 :                     Acrylic                MineralOil                         -                         - 
     15 : 16 :          LiquidScintillator                   Acrylic                         -                         - 
     16 :*17*:                     Acrylic        LiquidScintillator                         -                         - 
     17 : 18 :                   GdDopedLS                   Acrylic                         -                         - 

::

    In [116]: bb = aa[ls]

    In [119]: obb = np.arange(len(bb))[bb[:,0,0] != bb[:,0,2]]

    In [121]: len(obb)
    Out[121]: 419
    
::

    In [123]: float(len(obb))/len(bb)
    Out[123]: 0.004409783615391092


    
First boundaries of Ac gensteps
--------------------------------

::

    In [126]: acb = count_unique(aa[ac][:,0,3])
    Out[126]: 
    array([[ -25,  125],
           [ -23,    3],
           [ -20,   11],
           [ -18, 1977],
           [ -17,  137],
           [ -16, 1699],
           [ -15,   11],
           [  15, 1565],
           [  16,  103],
           [  17,  791]])

    In [173]: bnd = bnd_()

    In [168]: bd_ = lambda _:bnd[_]

    In [174]: map(bd_, acb[:,0] )
    Out[174]: 
    [u'(-25) Acrylic/MineralOil/-/RSOilSurface ',
     u'(-23) UnstStainlessSteel/MineralOil/-/- ',
     u'(-20) Pyrex/MineralOil/-/- ',
     u'(-18) GdDopedLS/Acrylic/-/- ',
     u'(-17) Acrylic/LiquidScintillator/-/- ',
     u'(-16) LiquidScintillator/Acrylic/-/- ',
     u'(-15) Acrylic/MineralOil/-/- ',
     u'(+15) Acrylic/MineralOil/-/- ',
     u'(+16) LiquidScintillator/Acrylic/-/- ',
     u'(+17) Acrylic/LiquidScintillator/-/- ']
     

Observations from Aux buffer : boundary 0 (MISS)
-------------------------------------------------

::

    In [1]: a = auc_(1).reshape(-1,10,4)
    INFO:env.numerics.npy.types:loading /usr/local/env/dayabay/aucerenkov/1.npy 
    -rw-r--r--  1 blyth  staff  49027360 Sep 19 11:59 /usr/local/env/dayabay/aucerenkov/1.npy

    In [2]: a
    Out[2]: 
    array([[[  6,   6,   0, -12],
            [  6,  12,   4, -13],
            [ 12,   4,   0, -14],
            ..., 
            [  0,   0,   0,   0],
            [  0,   0,   0,   0],
            [  0,   0,   0,   0]],


    In [7]: im = imat_()
    INFO:env.numerics.npy.types:parsing json for key ~/.opticks/GMaterialIndexLocal.json

    In [11]: map(lambda _:im[_], [6,6,6,12,12,4] )
    Out[11]: 
    ['IwsWater',
     'IwsWater',
     'IwsWater',
     'StainlessSteel',
     'StainlessSteel',
     'MineralOil']







What could go wrong ? The normal ?
------------------------------------

Winding order of vertex indices in indexBuffer determines the sidedness
of the triangle (ie in/out direction of the normal).

TODO: duplicate normal calc in geometry shader to visualize the normals


env/graphics/optixrap/cu/TriangleMesh.cu::

     34 RT_PROGRAM void mesh_intersect(int primIdx)
     35 {
     36     int3 index = indexBuffer[primIdx];
     37 
     38     //  tried flipping vertex order in unsuccessful attempt to 
     39     //  get normal shader colors to match OpenGL
     40     //  observe with touch mode that n.z often small
     41     //  ... this is just because surfaces are very often vertical
     42     //
     43     float3 p0 = vertexBuffer[index.x];
     44     float3 p1 = vertexBuffer[index.y];
     45     float3 p2 = vertexBuffer[index.z];
     46 
     47     float3 n;
     48     float  t, beta, gamma;
     49     if(intersect_triangle(ray, p0, p1, p2, n, t, beta, gamma))
     50     {
     51         if(rtPotentialIntersection( t ))
     52         {
     53             // attributes should be set between rtPotential and rtReport
     54             geometricNormal = normalize(n);
     55 
     56             nodeIndex = nodeBuffer[primIdx];
     57             boundaryIndex = boundaryBuffer[primIdx];
     58             sensorIndex = sensorBuffer[primIdx];
     59 
     60             // doing calculation here might (depending on OptiX compiler cleverness) 
     61             // repeat for all intersections encountered unnecessarily   
     62             // instead move the calculation into closest_hit
     63             //
     64             // intersectionPosition = ray.origin + t*ray.direction  ; 
     65             // http://en.wikipedia.org/wiki/Line–plane_intersection
     66 
     67             rtReportIntersection(0);




/Developer/OptiX/include/optixu/optixu_math_namespace.h::

    2022 /** Intersect ray with CCW wound triangle.  Returns non-normalize normal vector. */
    2023 OPTIXU_INLINE RT_HOSTDEVICE bool intersect_triangle(const Ray&    ray,
    2024                                                     const float3& p0,
    2025                                                     const float3& p1,
    2026                                                     const float3& p2,
    2027                                                           float3& n,
    2028                                                           float&  t,
    2029                                                           float&  beta,
    2030                                                           float&  gamma)
    2031 {
    2032   return intersect_triangle_branchless(ray, p0, p1, p2, n, t, beta, gamma);
    2033 }
    ....
    1956 /** Branchless intesection avoids divergence.
    1957 */
    1958 OPTIXU_INLINE RT_HOSTDEVICE bool intersect_triangle_branchless(const Ray&    ray,
    1959                                                                const float3& p0,
    1960                                                                const float3& p1,
    1961                                                                const float3& p2,
    1962                                                                      float3& n,
    1963                                                                      float&  t,
    1964                                                                      float&  beta,
    1965                                                                      float&  gamma)
    1966 {
    1967   const float3 e0 = p1 - p0;
    1968   const float3 e1 = p0 - p2;
    1969   n  = cross( e1, e0 );
    1970 
    1971   const float3 e2 = ( 1.0f / dot( n, ray.direction ) ) * ( p0 - ray.origin );
    1972   const float3 i  = cross( ray.direction, e2 );
    1973 
    1974   beta  = dot( i, e1 );
    1975   gamma = dot( i, e0 );
    1976   t     = dot( n, e2 );
    1977 
    1978   return ( (t<ray.tmax) & (t>ray.tmin) & (beta>=0.0f) & (gamma>=0.0f) & (beta+gamma<=1) );
    1979 }




"""

import sys
import numpy as np
from env.numerics.npy.types import phc_, iflags_, imat_, iabmat_, ihex_, seqhis_, seqmat_, stc_, c2g_


def sequence(h,a,b, gm, qm):
    #np.set_printoptions(formatter={'int':lambda x:hex(int(x))})
   
    im = imat_()

    for i,p in enumerate(h[a:b]):
        ix = a+i
        his,mat = p[0],p[1]
        print " %7d : %16s %16s : %30s %30s : %10s %10s " % ( ix,ihex_(his),ihex_(mat), 
                   seqhis_(his), seqmat_(mat),
                   im[gm[ix]], im[qm[ix]] 
                  )    


if __name__ == '__main__':

    tag = 1 

    gs = stc_(tag)                                       # gensteps 
    seq = phc_(tag).reshape(-1,2)                        # flag,material sequence
    c2g = c2g_()                                      # chroma index to ggeo custom int
    im = iabmat_()

    gsnph = gs.view(np.int32)[:,0,3]                     # genstep: num photons 
    gspdg = gs.view(np.int32)[:,3,0]

    gsmat_c = gs.view(np.int32)[:,0,2]                     # genstep: material code  : in chroma material map lingo
    gsmat   = np.array( map(lambda _:c2g.get(_,-1), gsmat_c ), dtype=np.int32)   # translate chroma indices into ggeo custom ones


    p_gsmat   = np.repeat(gsmat, gsnph)                    # photon: genstep material code  
    p_seqmat = seq[:,1] & 0xF                             # first material in seqmat 
    off = np.arange(len(p_gsmat))[ p_gsmat != p_seqmat ]

    p_gsidx = np.repeat(np.arange(len(gsnph)), gsnph )    # photon: genstep index


    #n = len(sys.argv)
    #a = int(sys.argv[1]) if n > 1 else 0
    #b = int(sys.argv[2]) if n > 2 else a + 40
    #
    #print n,a,b
    #sequence(seq,a,b, p_gsmat, p_seqmat)

    for i in off:
        his,mat = seq[i,0],seq[i,1]
        seqhis = seqhis_(his)
        if 'MI' in seqhis:continue
        print " %7d : %16s %16s : %30s %30s : gs:%2s sq:%2s " % ( i,ihex_(his),ihex_(mat), 
                   seqhis, seqmat_(mat),
                   im[p_gsmat[i]], im[p_seqmat[i]] 
                  )    




   




