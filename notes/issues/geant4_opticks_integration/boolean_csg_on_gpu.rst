Boolean CSG on GPU
===================

TODO: boolean trees implementation
------------------------------------

TODO: numerical/chi2 history comparison with CFG4 booleans 
------------------------------------------------------------



FIXED Issue : boolean intersection "lens" : boundary disappears from inside
------------------------------------------------------------------------------

**FIXED by starting tmin from propagate_epsilon, as during propagation photons start on boundaries**


Using boolean sphere-sphere intersection to construct a lens.::

     72 tboolean-testconfig()
     73 {
     74     local material=GlassSchottF2
     75     #local material=MainH2OHale
     76 
     77     local test_config=(
     78                  mode=BoxInBox
     79                  analytic=1
     80 
     81                  shape=box      parameters=0,0,0,1200               boundary=Rock//perfectAbsorbSurface/Vacuum
     82 
     83                  shape=intersection parameters=0,0,0,400            boundary=Vacuum///$material
     84                  shape=sphere       parameters=0,0,-600,641.2          boundary=Vacuum///$material
     85                  shape=sphere       parameters=0,0,600,641.2           boundary=Vacuum///$material
     86 
     87                )
     91      echo "$(join _ ${test_config[@]})" 
     92 }

Observe that photons reflecting inside the lens off the 2nd boundary do 
not intersect with the 1st boundary on their way back yielding "TO BT BR SA"

Similarly, and more directly, also have "TO BT SA" not seeing the 2nd boundary. 

Initially thought the raytrace confirmed this as 
it looked OK from outside but when go inside the boundary disappears, but
that turns out to be just near clipping.

::

    tboolean-;tboolean--




FIXED Issue : lens not bending light 
--------------------------------------

Fixed by passing the boundary index 
via the instanceIdentity attribute from intersection 
to closest hit progs.


approach
-----------


ggeo/GPmt.hh
ggeo/GCSG.hh
    Brings python prepared CSG tree for DYB PMT into GPmt member

    Looks like GCSG is currently being translated into into 
    partBuffer/solidBuffer representation prior to GPU ? 




hemi-pmt.cu::

    /// flag needed in solidBuffer
    ///
    ///   0:primitive
    ///   1:boolean-intersect
    ///   2:boolean-union
    ///   3:boolean-difference
    ///
    /// presumably the numParts will be 2 for booleans
    /// thence can do the sub-intersects and boolean logic
    /// 
    /// ...
    /// need to elide the sub-solids from OptiX just passing booleans
    /// in as a single solidBuffer entry with numParts = 2 ?
    ///
    /// maybe change name solidBuffer->primBuffer
    /// as booleans handled as OptiX primitives composed of two parts
    ///   

    1243 RT_PROGRAM void intersect(int primIdx)
    1244 {
    1245   const uint4& solid    = solidBuffer[primIdx];
    1246   unsigned int numParts = solid.y ;
    ....
    1252   uint4 identity = identityBuffer[instance_index] ;
    1254 
    1255   for(unsigned int p=0 ; p < numParts ; p++)
    1256   {
    1257       unsigned int partIdx = solid.x + p ;
    1258 
    1259       quad q0, q1, q2, q3 ;
    1260 
    1261       q0.f = partBuffer[4*partIdx+0];
    1262       q1.f = partBuffer[4*partIdx+1];
    1263       q2.f = partBuffer[4*partIdx+2] ;
    1264       q3.f = partBuffer[4*partIdx+3];
    1265 
    1266       identity.z = q1.u.z ;  // boundary from partBuffer (see ggeo-/GPmt)
    1267 
    1268       int partType = q2.i.w ;
    1269 
    1270       // TODO: use enum      
    ////     this is the NPart.hpp enum 
    ////
    1271       switch(partType)
    1272       {
    1273           case 0:
    1274                 intersect_aabb(q2, q3, identity);
    1275                 break ;
    1276           case 1:
    1277                 intersect_zsphere<false>(q0,q1,q2,q3,identity);
    1278                 break ;



