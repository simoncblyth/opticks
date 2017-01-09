Boolean CSG on GPU
===================

TODO: boolean trees implementation
------------------------------------

TODO: numerical/chi2 history comparison with CFG4 booleans 
------------------------------------------------------------

Issue : ray trace of box shows slab intersects extending behind the box
--------------------------------------------------------------------------

Presumably infinity or NaN handling.

How to debug ?
~~~~~~~~~~~~~~~

* checked the non-boolean box, thats working fine with no artifacts.

* Using discaxial torch type to shoot photons from 26 positions 
  and directions, so can feel the geometry in a numerical manner.

* when on target, things look correct, the same as the non-boolen box
  when off target the invalid intersects manifest 


::

    local discaxial_hit=0,0,0
    local discaxial_miss=0,0,300
    local torch_config_discaxial=(
                 type=discaxial
                 photons=$photons
                 frame=-1
                 transform=$identity
                 source=$discaxial_hit
                 target=0,0,0
                 time=0.1
                 radius=110
                 distance=200
                 zenithazimuth=0,1,0,1
                 material=Vacuum
                 wavelength=$wavelength
               )


Axis aligned photon directions appear to be part of the problem at least::

    421       else if( ts.type == T_DISCAXIAL )
    422       {
    423           unsigned long long photon_id = launch_index.x ;
    424 
    425           //float3 dir = get_direction_26( photon_id % 26 );
    426           //float3 dir = get_direction_6( photon_id % 6 );
    427           //float3 dir = get_direction_6( photon_id % 4, -0.00001f );  // 1st 4: +X,-X,+Y,-Y   SPURIOUS INTERSECTS GONE
    428           //float3 dir = get_direction_6( photon_id % 4, -0.f );       // 1st 4: +X,-X,+Y,-Y   SPURIOUS INTERSECTS GONE
    429           float3 dir = get_direction_6( photon_id % 4, 0.f );          // 1st 4: +X,-X,+Y,-Y   SPURIOUS INTERSECTS BACK AGAIN
    430           
    431           float r = radius*sqrtf(u1) ; // sqrt avoids pole bunchung  
    432           float3 discPosition = make_float3( r*cosPhi, r*sinPhi, 0.f );
    433           rotateUz(discPosition, dir);
    434           
    435           // ts.x0 should be placed inside the target when hits are desired
    436           // wih DISCAXIAL mode
    437           p.position = ts.x0 + distance*dir + discPosition ;
    438           p.direction = -dir ;
    439           


Curious the direction zeros are all negative 0 resulting in -inf for both -X and +X directions::

  ray.origin 200.000000 -11.247929 307.520966 ray.direction -1.000000 -0.000000 -0.000000 idir -1.000000 -inf -inf 
  ray.origin 200.000000 44.386002 262.619629 ray.direction -1.000000 -0.000000 -0.000000 idir -1.000000 -inf -inf 
  ray.origin 200.000000 -88.033470 321.681213 ray.direction -1.000000 -0.000000 -0.000000 idir -1.000000 -inf -inf 
  ray.origin 200.000000 -39.863480 244.735748 ray.direction -1.000000 -0.000000 -0.000000 idir -1.000000 -inf -inf 
  ray.origin -200.000000 97.620598 274.010651 ray.direction 1.000000 -0.000000 -0.000000 idir 1.000000 -inf -inf 
  ray.origin 200.000000 8.609403 199.297638 ray.direction -1.000000 -0.000000 -0.000000 idir -1.000000 -inf -inf 
  ray.origin -200.000000 -67.498100 266.557739 ray.direction 1.000000 -0.000000 -0.000000 idir 1.000000 -inf -inf 
  ray.origin -200.000000 78.251770 366.333496 ray.direction 1.000000 -0.000000 -0.000000 idir 1.000000 -inf -inf 
  ray.origin -200.000000 47.188507 215.060699 ray.direction 1.000000 -0.000000 -0.000000 idir 1.000000 -inf -inf 

Using a delta 0.00001f get -1/delta and spurious interects remain::

  ray.origin 200.000778 9.482430 213.216736 ray.direction -1.000000 -0.000010 -0.000010 idir -1.000000 -100000.000000 -100000.000000 
  ray.origin -199.999054 48.094410 346.568787 ray.direction 1.000000 -0.000010 -0.000010 idir 1.000000 -100000.000000 -100000.000000 

Bizarrely switching to delta -0.00001f get 1/delta and the spurious intersects are gone::

  ray.origin 199.999344 -88.035469 321.679199 ray.direction -1.000000 0.000010 0.000010 idir -1.000000 100000.000000 100000.000000 
  ray.origin 199.999222 9.478431 213.212708 ray.direction -1.000000 0.000010 0.000010 idir -1.000000 100000.000000 100000.000000 
  ray.origin 200.000000 49.761848 249.952194 ray.direction -1.000000 0.000010 0.000010 idir -1.000000 100000.000000 100000.000000 
  ray.origin 200.000748 39.745564 334.747955 ray.direction -1.000000 0.000010 0.000010 idir -1.000000 100000.000000 100000.000000 
  ray.origin -199.999298 -8.694067 238.793365 ray.direction 1.000000 0.000010 0.000010 idir 1.000000 100000.000000 100000.000000 
  ray.origin 199.999878 -76.475029 363.946503 ray.direction -1.000000 0.000010 0.000010 idir -1.000000 100000.000000 100000.000000 
  ray.origin 200.000290 44.076099 285.449768 ray.direction -1.000000 0.000010 0.000010 idir -1.000000 100000.000000 100000.000000 

Same when using -0.f::

    425           //float3 dir = get_direction_26( photon_id % 26 );
    426           //float3 dir = get_direction_6( photon_id % 6 );
    427           //float3 dir = get_direction_6( photon_id % 4, -0.00001f );     // 1st 4: +X,-X,+Y,-Y 
    428           float3 dir = get_direction_6( photon_id % 4, -0.f );     // 1st 4: +X,-X,+Y,-Y 
    429           
    430           float r = radius*sqrtf(u1) ; // sqrt avoids pole bunchung  
    431           float3 discPosition = make_float3( r*cosPhi, r*sinPhi, 0.f );
    432           rotateUz(discPosition, dir);
    433           
    434           // ts.x0 should be placed inside the target when hits are desired
    435           // wih DISCAXIAL mode
    436           p.position = ts.x0 + distance*dir + discPosition ;
    437           p.direction = -dir ;

::

  ray.origin 200.000000 14.684715 244.904205 ray.direction -1.000000 0.000000 0.000000 idir -1.000000 inf inf 
  ray.origin 200.000000 -68.328766 251.635269 ray.direction -1.000000 0.000000 0.000000 idir -1.000000 inf inf 
  ray.origin -200.000000 102.468193 335.907471 ray.direction 1.000000 0.000000 0.000000 idir 1.000000 inf inf 
  ray.origin 200.000000 -26.478765 307.570923 ray.direction -1.000000 0.000000 0.000000 idir -1.000000 inf inf 
  ray.origin 200.000000 -15.085106 304.063721 ray.direction -1.000000 0.000000 0.000000 idir -1.000000 inf inf 


::

     42    float3 idir = make_float3(1.f)/ray.direction ;
     43    float3 t0 = (bmin - ray.origin)*idir;
     44    float3 t1 = (bmax - ray.origin)*idir;


::

     idir -1.000000 -inf -inf t0 300.000000 inf inf t1 100.000000 -inf inf 
     idir -1.000000 -inf -inf t0 300.000000 inf inf t1 100.000000 -inf inf 
     idir -1.000000 -inf -inf t0 300.000000 inf inf t1 100.000000 -inf inf 
     idir -1.000000 -inf -inf t0 300.000000 inf inf t1 100.000000 -inf inf 
     idir 1.000000  -inf -inf t0 100.000000 inf inf t1 300.000000 -inf inf 
     idir 1.000000  -inf -inf t0 100.000000 inf inf t1 300.000000 -inf inf 
     idir 1.000000  -inf -inf t0 100.000000 inf inf t1 300.000000 -inf inf 
     idir 1.000000  -inf -inf t0 100.000000 inf inf t1 300.000000 -inf inf 





CUDA fminf/fmaxf/max infinity/nan handling ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

::

    simon:include blyth$ grep fminf *.*
    device_functions.h:__DEVICE_FUNCTIONS_STATIC_DECL__ float fminf(float x, float y);
    device_functions.hpp:__DEVICE_FUNCTIONS_STATIC_DECL__ float fminf(float x, float y)
    device_functions.hpp:  return __nv_fminf(x, y);
    device_functions_decls.h:__DEVICE_FUNCTIONS_DECLS__ float __nv_fminf(float x, float y);
    math_functions.h:extern __host__ __device__ __device_builtin__ float                  fminf(float x, float y) __THROW;
    math_functions.h:extern __host__ __device__ __device_builtin__ _CRTIMP float  __cdecl fminf(float x, float y);
    math_functions.h:__func__(float fminf(float a, float b));
    math_functions.hpp:  return fminf(a, b);
    math_functions.hpp:  return fminf(a, b);
    math_functions.hpp:__func__(float fminf(float a, float b))
    nppi_color_conversion.h: *  This code uses the fmaxf() and fminf() 32 bit floating point math functions.
    nppi_color_conversion.h: *  Npp32f nMin = fminf(nNormalizedR, nNormalizedG);
    nppi_color_conversion.h: *         nMin = fminf(nMin, nNormalizedB);
    nppi_color_conversion.h: *  This code uses the fmaxf() and fminf() 32 bit floating point math functions.
    nppi_color_conversion.h: *  Npp32f nTemp = fminf(nNormalizedR, nNormalizedG);
    nppi_color_conversion.h: *         nTemp = fminf(nTemp, nNormalizedB);
    simon:include blyth$ 
    simon:include blyth$ 
    simon:include blyth$ pwd
    /Developer/NVIDIA/CUDA-7.0/include





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



