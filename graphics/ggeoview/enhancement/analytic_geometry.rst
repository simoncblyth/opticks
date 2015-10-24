Analytic Geometry
===================


Principal
----------

Triangle meshes are a convenient initial step to the GPU 
as all geometry can be treated with the same code.
Special treatment of important geometry (PMTs) however
is expected to have large performance gains.

Ray intersection with CSG solids boils down to 
analytic solving quadratic/cubic polynomials. There is 
a technique to handle union intersections by applying boolean operations
to intersection segments of the sub volumes. 


Face Slicing
-------------

::

   ggv-pmt --fslice 0:720
   ggv-pmt --fslice 720:1392
   ggv-pmt --fslice 1392:2352
   ggv-pmt --fslice 2352:2832
   ggv-pmt --fslice 2832:2928

       # selecting faces of single solids, nodeinfo.npy provides the face index ranges 

::

    In [1]: ni = np.load("GMergedMesh/1/nodeinfo.npy")

    In [2]: ni
    Out[2]: 
    array([[ 720,  362, 3199, 3155],
           [ 672,  338, 3200, 3199],
           [ 960,  482, 3201, 3200],
           [ 480,  242, 3202, 3200],
           [  96,   50, 3203, 3200]], dtype=uint32)

    In [3]: np.cumsum(ni[:,0])
    Out[3]: array([ 720, 1392, 2352, 2832, 2928], dtype=uint64)


Photocathode
-------------

::

    pmt-parts   # move to writing full partition file, and pslicing as needed

    ggv-pmt --fslice 1392:2352 --pslice 8:10



First and Second Solids, Pyrex and contained vacuum
-------------------------------------------------------

OptiX render is as would expect, with pyrex and vacuum very thinly separated, 
to make the inner volume visible adjust near to control the ray trace epsilon

OpenGL render not as would expect, much fatter to the back. 
As if pushed out by the dynode ?

::

   pmt-parts 0:8 
   ggv-pmt --fslice 0:1392



Tubs Issue FIXED, was caused by cylinder poking outside its bbox
-------------------------------------------------------------------

* enable ENDCAP_P only in pmt-/dd.py and regen with::

  pmt-parts 3:4

* setup coloring in cu/pinhole_camera.cu::

   100   // BGRA
   101   uchar4 color = prd.flag == HP_PCAP_I ? RED :  make_color( prd.result );


* get expected behavior for outer and inner HP_PCAP_O and HP_PCAP_I

* PCAP endcap is to the right(in default initial ggv-pmt viewpoint) 

* doing the same for QCAP see view dependent shape mis-behaviour, but disabling the 
  partition_union resetting of bbox avoids it

* the problem was the bbox was clipped in at the 3spehere interseciton plane 
  but ZSize was not changed

* from point of view of cylinder rendering the relevant PQ vector is not (0,0,sizeZ)
  but rather (0,0,clipped_sizeZ)

::

    194 static __device__
    195 void intersect_ztubs(quad& q0, quad& q1, quad& q2, quad& q3, const uint4& identity )
    196 {
    197 /* 
    198 Position shift below is to match between different cylinder Z origin conventions
    199 
    200 * Ericson calc implemented below has cylinder origin at endcap P  
    201 * detdesc/G4 Tubs has cylinder origin in the center 
    202 
    203 */
    204     float sizeZ = q1.f.x ;
    205     float z0 = q0.f.z - sizeZ/2.f ;
    206     float3 position = make_float3( q0.f.x, q0.f.y, z0 );  // 0,0,-169.
    207     float clipped_sizeZ = q3.f.z - q2.f.z ;
    208 
    209     float radius = q0.f.w ;
    210     int flags = q1.i.w ;  
    211     
    212     bool PCAP = flags & ENDCAP_P ;
    213     bool QCAP = flags & ENDCAP_Q ;
    214     
    215     //rtPrintf("intersect_ztubs position %10.4f %10.4f %10.4f \n", position.x, position.y, position.z );
    216     //rtPrintf("intersect_ztubs flags %d PCAP %d QCAP %d \n", flags, PCAP, QCAP);
    217     
    218     float3 m = ray.origin - position ;
    219     float3 n = ray.direction ; 
    220     float3 d = make_float3(0.f, 0.f, clipped_sizeZ );
    221     
    222     float rr = radius*radius ;
    223     float3 dnorm = normalize(d);
    224     





   


Just Tubs
----------

Some funny straight lines as rotate around::

   pmt-parts 3:4   # just tubs

   ggv-pmt 

Either a bug or maybe optical illusion due to:

* perspective projection 
* no depth/inside/outside queues 

Perhaps Z cut happening in wrong frame ? 

TODO:

* get orthographic projection working for OptiX raygen 
* matplotlib projection plot of points of the mesh 

::




    In [4]: v = np.load("GMergedMesh/1/vertices.npy")

    In [5]: v
    Out[5]: 
    array([[   0.   ,    0.   ,  131.   ],
           [  33.905,    0.   ,  126.536],
           [  32.75 ,    8.775,  126.536],
           ..., 
           [  26.563,   -7.118,    1.5  ],
           [   0.   ,    0.   ,    1.5  ],
           [   0.   ,    0.   , -164.5  ]], dtype=float32)

    In [6]: v.shape
    Out[6]: (1474, 3)

    In [7]: ni[:,1].sum()  ## sum of vertices, it matches as these are fixed meshes with no dupes
    Out[7]: 1474


    In [10]: i = np.load("GMergedMesh/1/indices.npy").reshape(-1,3)

    In [11]: i.shape
    Out[11]: (2928, 3)

    In [15]: np.unique(i[:720]).min()
    Out[15]: 0

    In [16]: np.unique(i[:720]).max()
    Out[16]: 361

    n [12]: ni[:,0].sum()
    Out[12]: 2928

    In [19]: np.unique(i[:720]).size    # hmm no need for doing indices look up into the vertices, its all contiguous
    Out[19]: 362



Just Tracing a single instance
--------------------------------

Using OTracerTest with the below is much faster than with 
full context (including all those propagate buffers) and full geometry::

   pmt-parts 0:4   # 3sphere + tubs


   ggv --tracer --restrictmesh 1 --analyticmesh 1 --islice 0 --target 3199

   ggv-pmt    # abbreviation for above

   ggv-allpmt --stack $((1024 + 512))      # stack can be reduced a bit with just the tracer


   ggv --tracer --restrictmesh 1 --analyticmesh 1 
    
   ggv-allpmt 



Plumbing check
----------------

::

    ggv --restrictmesh 1 --analyticmesh 1 --torchconfig "radius=300;frame=3199;source=0,0,1000;target=0,0,0"


How to OptiX intersect with CSG solid ?
-----------------------------------------
::

    simon:OptiX_380_sdk blyth$ find . -name '*.cu'  -exec grep -l intersect {} \;
    ./ambocc/parallelogram.cu
    ./ambocc/sphere.cu
    ./buffersOfBuffers/parallelogram.cu
    ./buffersOfBuffers/sphere_texcoord.cu
    ./cook/clearcoat.cu
    ./cook/dof_camera.cu
    ./cook/parallelogram.cu
    ./cook/sphere.cu
    ./cook/sphere_texcoord.cu
    ./cuda/triangle_mesh.cu
    ./cuda/triangle_mesh_small.cu
    ./device_exceptions/device_exceptions.cu
    ./displacement/geometry_programs.cu
    ./glass/glass.cu
    ./glass/triangle_mesh_iterative.cu
    ./heightfield/heightfield.cu
    ./hybridShadows/triangle_mesh_fat.cu
    ./isgReflections/parallelogram.cu
    ./isgReflections/triangle_mesh_fat.cu
    ./isgShadows/triangle_mesh_fat.cu
    ./julia/block_floor.cu
    ./julia/julia.cu
    ...

    simon:OptiX_380_sdk blyth$ find . -type f -exec grep -l union {} \;
    ./julia/block_floor.cu
    ./julia/distance_field.h


Julia sample has lots of non-trivial intersection examples


julia/block_floor.cu::

    538 RT_PROGRAM void intersect(int primIdx)
    539 {
    540   object_factory<false>::Object obj;
    541   object_factory<false>::make_object(obj, ray.direction);
    542 
    543   // first check for intersection between the ray and aabb
    544   optix::Ray tmp_ray = ray;
    545   if(intersect_aabb(tmp_ray, obj)) {
    546     float epsilon = 1.25e-3f;
    547     float max_epsilon = 2.5e-2f;
    548 
    549     float3 hit_point;
    550     float t = adaptive_sphere_trace<1000>(tmp_ray, make_distance_to_primitive(obj), hit_point, epsilon, max_epsilon);
    551     if(t < tmp_ray.tmax)
    552     {
    553       if(rtPotentialIntersection(t))

 
julia/distance_field.h::

    216 // The union of two primitives
    217 template<typename Primitive1, typename Primitive2>
    218   class PrimitiveUnion
    219 {
    220   public:
    221     // null constructor creates an undefined DistanceUnion
    222     HD_DECL
    223     PrimitiveUnion(void){}
    224 
    225     HD_DECL
    226     PrimitiveUnion(Primitive1 p1, Primitive2 p2):m_prim1(p1),m_prim2(p2){}
    227 
    228     HD_DECL
    229     float distance(const float3 &x) const
    230     {
    231       return fminf(m_prim1.distance(x), m_prim2.distance(x));
    232     }
    ...
      


shadeTree/parallelogram.cu::

     37 RT_PROGRAM void intersect(int primIdx)
     38 {
     39   float3 n = make_float3( plane );
     40   float dt = dot(ray.direction, n );
     41   float t = (plane.w - dot(n, ray.origin))/dt;
     42   if( t > ray.tmin && t < ray.tmax ) {
     43     float3 p = ray.origin + ray.direction * t;
     44     float3 vi = p - anchor;
     45     float a1 = dot(v1, vi);
     46     if(a1 >= 0 && a1 <= 1){
     47       float a2 = dot(v2, vi);
     48       if(a2 >= 0 && a2 <= 1){
     49         if( rtPotentialIntersection( t ) ) {
     50           geometric_normal = n;
     51           shading_normal = n;
     52           uv = make_float2(a1, a2);
     53           rtReportIntersection( 0 );
     54         }
     55       }
     56     }
     57   }
     58 }


tutorial.cpp::

    238 float4 make_plane( float3 n, float3 p )
    239 {
    240   n = normalize(n);
    241   float d = -dot(n, p);
    242   return make_float4( n, d );
    243 }


tutorial10.cu::

    313 //
    314 // Intersection program for programmable convex hull primitive
    ///
    ///     https://en.wikipedia.org/wiki/Line–plane_intersection
    ///     http://geomalgorithms.com/index.html
    ///
    315 //
    316 rtBuffer<float4> planes;
    317 RT_PROGRAM void chull_intersect(int primIdx)
    318 {
    319   int n = planes.size();
    320   float t0 = -FLT_MAX;
    321   float t1 = FLT_MAX;
    322   float3 t0_normal = make_float3(0);
    323   float3 t1_normal = make_float3(0);
    324   for(int i = 0; i < n && t0 < t1; ++i ) {
    325     float4 plane = planes[i];
    326     float3 n = make_float3(plane);
    327     float  d = plane.w;
    328 
    329     float denom = dot(n, ray.direction);
    330     float t = -(d + dot(n, ray.origin))/denom;
    ///
    ///  Plane eqn, p0 is point in plane, n is normal 
    ///     (p - p0).n = 0
    ///
    ///  Line 
    ///      p = ray.origin + t * ray.direction
    ///
    ///  Intersect
    ///
    ///    (ray.origin + t * ray.direction - p0 ).n = 0 
    ///
    ///     dot(n, ray.origin) + t * dot(n, ray.direction) - dot(p0, n) = 0  
    ///                
    ///                  dot(p0,n) - dot(n, ray.origin)
    ///            t =  --------------------------------           
    ///                     dot(n, ray.direction)
    ///
    ///

    331     if( denom < 0){
    332       // enter
    333       if(t > t0){
    334         t0 = t;
    335         t0_normal = n;
    336       }
    337     } else {
    338       //exit
    339       if(t < t1){
    340         t1 = t;
    341         t1_normal = n;
    342       }
    343     }
    344   }
    345 
    346   if(t0 > t1)
    347     return;
    348 
    349   if(rtPotentialIntersection( t0 )){
    350     shading_normal = geometric_normal = t0_normal;
    351     rtReportIntersection(0);
    352   } else if(rtPotentialIntersection( t1 )){
    353     shading_normal = geometric_normal = t1_normal;
    354     rtReportIntersection(0);
    355   }
    356 }







How to proceed ?
------------------

* on revisiting G4DAE include GDML G4 CSG model description together
  with the triangulated COLLADA 


detdesc PMT is involved
------------------------

Complicated assemblies of CSG solids. Implementing analytic is non-trivial.

G5:/home/blyth/local/env/dyb/NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB/PMT/geometry.xml::

     08   <catalog name="PMT">
     09 
     10     <logvolref href="hemi-pmt.xml#lvPmtHemiFrame"/>
     11     <logvolref href="hemi-pmt.xml#lvPmtHemi"/>
     12     <logvolref href="hemi-pmt.xml#lvPmtHemiwPmtHolder"/>
     13     <logvolref href="hemi-pmt.xml#lvAdPmtCollar"/>
     14     <logvolref href="hemi-pmt.xml#lvPmtHemiCathode"/>
     15     <logvolref href="hemi-pmt.xml#lvPmtHemiVacuum"/>
     16     <logvolref href="hemi-pmt.xml#lvPmtHemiBottom"/>
     ..

dybgaudi/Detector/XmlDetDesc/DDDB/PMT/hemi-pmt.xml::

     37   <!-- The PMT glass -->
     38   <logvol name="lvPmtHemi" material="Pyrex">
     39     <union name="pmt-hemi">
     40       <intersection name="pmt-hemi-glass-bulb">
     41           <sphere name="pmt-hemi-face-glass"
     42                 outerRadius="PmtHemiFaceROC"/>
     43 
     44           <sphere name="pmt-hemi-top-glass"
     45                outerRadius="PmtHemiBellyROC"/>
     46           <posXYZ z="PmtHemiFaceOff-PmtHemiBellyOff"/>
     47 
     48           <sphere name="pmt-hemi-bot-glass"
     49                 outerRadius="PmtHemiBellyROC"/>
     50           <posXYZ z="PmtHemiFaceOff+PmtHemiBellyOff"/>
     51 
     52       </intersection>
     53       <tubs name="pmt-hemi-base"
     54         sizeZ="PmtHemiGlassBaseLength"
     55         outerRadius="PmtHemiGlassBaseRadius"/>
     56       <posXYZ z="-0.5*PmtHemiGlassBaseLength"/>
     57     </union>
     58 
     59     <physvol name="pvPmtHemiVacuum"
     60          logvol="/dd/Geometry/PMT/lvPmtHemiVacuum"/>
     61 
     62   </logvol>


::

    118   <!-- The Photo Cathode -->
    119   <!-- use if limit photocathode to a face on diameter gt 167mm. -->
    120   <logvol name="lvPmtHemiCathode" material="Bialkali" sensdet="DsPmtSensDet">
    121     <union name="pmt-hemi-cathode">
    122       <sphere name="pmt-hemi-cathode-face"
    123           outerRadius="PmtHemiFaceROCvac"
    124           innerRadius="PmtHemiFaceROCvac-PmtHemiCathodeThickness"
    125           deltaThetaAngle="PmtHemiFaceCathodeAngle"/>
    126       <sphere name="pmt-hemi-cathode-belly"
    127           outerRadius="PmtHemiBellyROCvac"
    128           innerRadius="PmtHemiBellyROCvac-PmtHemiCathodeThickness"
    129           startThetaAngle="PmtHemiBellyCathodeAngleStart"
    130           deltaThetaAngle="PmtHemiBellyCathodeAngleDelta"/>
    131       <posXYZ z="PmtHemiFaceOff-PmtHemiBellyOff"/>
    132     </union>
    133   </logvol>









