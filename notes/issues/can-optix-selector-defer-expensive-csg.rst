Can OptiX Selector Defer Expensive CSG ?
===========================================

Overview
---------

Ray tracing geometries composed of millions of triangles, or 
manually partitioned analytic geometries is substantially faster than
ray tracing CSG trees.

Perhaps CSG tree ray tracing can be deferred (or simplified) until actually needed by 
replacing the complicated CSG trees for PMT instances with just their
outer volumes for rays originating outside the bbox ? Or beyond some
distance from instance transform center.


Current Working 
------------------

::


    op --debugger --gltf 3  --tracer --rendermode +in0 --restrictmesh 5
          
          # just analytic PMTs in raytrace, just triangulated in OpenGL
          #
          # (required OpticksHub::configureGeometry fix to apply restrictmesh equally to tri and ana)


    op --debugger --gltf 0  --tracer --rendermode +in0 --restrictmesh 5
          
          # just triangulated PMTs in OpenGL and raytrace
          #
          # (required OGeo::convert fix to only OGeo::makeAnalyticGeometry  for gltf > 0 )


    op --raylod --debugger --gltf 3  --tracer --rendermode +in0 --restrictmesh 5

          # just analytic PMTs in raytrace, just triangulated in OpenGL 
          #
          # --raylod (raytrace LOD) 
          #       causes a simplified analytic geometry (just outer volume) 
          #       to be used for rays with instance OBJECT frame length(ray.origin) > instance_bounding_radius      
          #
          #       In order to see any difference you need to position viewpoint close to some PMT
          #       instances and use near scanning(*n* then swipe up/down) to cut into the volume, 
          #       then vary position (*z* or *x*) and repeat the near scanning.  
          #       Only when very close should the multiple volumes of PMT innards be visible.
          #  


HMM : perhaps should enable OpenGL LOD in parallel to raytrace LOD ?
------------------------------------------------------------------------

Actually, need to revisit OpenGL LOD.  The raytrace way of making simplified geometry
from the outer volume is better than the bbox LOD levels.



DONE : outer volume only geometry for lod level 1
----------------------------------------------------

* used lazy LOD approach of not changing buffers, just ignoring most of them for
  LOD level 1 


Making triangulated OptiX geometry from mm uses pre-combined buffers, ie all 
the solids data are already concatenated into single buffers, the data accessed with.::

    mm->getVerticesBuffer()
    mm->getIndicesBuffer()
    mm->getFaceRepeatedInstancedIdentityBuffer()  OR mm->getFaceRepeatedIdentityBuffer()
 
Similarly for analytic OptiX geometry with the data already combined via GParts addition.::

     537 void GMergedMesh::mergeSolidAnalytic( GParts* pts, GMatrixF* transform, unsigned verbosity )
     538 {
     539     // analytic CSG combined at node level  
     ...
     547     if(transform && !transform->isIdentity())
     548     {
     549         pts->applyPlacementTransform(transform, verbosity );
     550     }
     551 
     552     if(!m_parts) m_parts = new GParts() ;
     553     m_parts->add(pts, verbosity);
     554 }


Hmm, can the 1st solids info be fished out of the combined ?

* hmm maybe trivial for analytic, just set numPrim to 1 ?
* for triangulated maybe just need the facecount for 1st prim 



FIXED : ice cream cone partial PMT instance geometry
---------------------------------------------------------
          
This ice cream cone issue turned out to be due to incorrect sharing of an 
acceleration structure between the ana and tri versions of the geometry.
          
* Can only share accel for exactly the same geometry.
          

::

    op --raylod --debugger --gltf 3  --tracer --rendermode +in0 --restrictmesh 5

          # trying to check if the visit_instance selector really kicks in 
          #  ... getting a bizarre little ice cream cone just at PMT cathode position, 
          # all at level 1
          #
          # looking closer it seems to be a quadrant of a fan from the front of the triangulated
          # PMT ... switching to dump only on getting a level 0, see none even when trying 
          # to navigate real close 


UNREPRODUCIBLE CRASH  : GPU contension assumed
--------------------------------------------------

* putting the crash down to Chrome GPU contention, moral: runs tests with minimal apps running, 
  especially avoid Safari, Chrome, Mail

Changing cut to 12m, starts OK with tri PMTs then crashes on getting close.  
This forced a reboot, relaunching, Chrome comes up in a stuck state, requiring force kill.
Possibly there was GPU contention coming from Chrome ?


FIXED : All instances changing at once
-------------------------------------------

Rerunning with only (Finder+Terminal) observe from a distance the broken triangulated fans, as get close, all instances change at once to the 
normal analytic PMTs, backing away the broken tri fans all reappear at once...

The all-at-once identified to be due to overwrite of instance position on the visit prog, all instances
had the position of the last instance.

::

    2017-10-17 20:24:59.254 INFO  [31358] [OContext::launch@322] OContext::launch LAUNCH time: 0.216804
    visit_instance: level 0 size  12000.000 distance  11889.045    origin (-19308.762 -806742.875  -6574.506) instance_position (-17951.662 -795436.938  -3156.400  12000.000)  
    visit_instance: level 0 size  12000.000 distance  11889.045    origin (-19308.762 -806742.875  -6574.506) instance_position (-17951.662 -795436.938  -3156.400  12000.000)  
    visit_instance: level 0 size  12000.000 distance  11889.045    origin (-19308.762 -806742.875  -6574.506) instance_position (-17951.662 -795436.938  -3156.400  12000.000)  
    visit_instance: level 0 size  12000.000 distance  11889.040    origin (-19308.715 -806742.875  -6574.506) instance_position (-17951.662 -795436.938  -3156.400  12000.000)  
    visit_instance: level 0 size  12000.000 distance  11889.045    origin (-19308.762 -806742.875  -6574.506) instance_position (-17951.662 -795436.938  -3156.400  12000.000)  
    visit_instance: level 0 size  12000.000 distance  11889.040    origin (-19308.715 -806742.875  -6574.506) instance_position (-17951.662 -795436.938  -3156.400  12000.000)  
    visit_instance: level 0 size  12000.000 distance  11889.040    origin (-19308.715 -806742.875  -6574.506) instance_position (-17951.662 -795436.938  -3156.400  12000.000)  
    visit_instance: level 0 size  12000.000 distance  11889.040    origin (-19308.715 -806742.875  -6574.506) instance_position (-17951.662 -795436.938  -3156.400  12000.000)  
    visit_instance: level 0 size  12000.000 distance  11889.045    origin (-19308.762 -806742.875  -6574.506) instance_position (-17951.662 -795436.938  -3156.400  12000.000)  

::

    [ 5] (     0/   672 )         ip-16572.898 -801469.625 -8842.500   1.000 
    [ 5] (     1/   672 )         ip-16166.072 -801019.375 -8842.500   1.000 
    [ 5] (     2/   672 )         ip-15889.641 -800479.188 -8842.500   1.000 
    [ 5] (     3/   672 )         ip-15762.440 -799885.875 -8842.500   1.000 
    [ 5] (     4/   672 )         ip-15793.142 -799279.812 -8842.500   1.000 
    [ 5] (   667/   672 )         ip-17951.662 -795436.938 -8458.400   1.000 
    [ 5] (   668/   672 )         ip-16627.082 -795720.875 -6128.400   1.000 
    [ 5] (   669/   672 )         ip-17951.662 -795436.938 -6128.400   1.000 
    [ 5] (   670/   672 )         ip-16627.082 -795720.875 -3156.400   1.000 
    [ 5] (   671/   672 )         ip-17951.662 -795436.938 -3156.400   1.000 
    2017-10-17 20:24:32.463 INFO  [31358] [SLog::operator@15] OScene::OScene DONE





FIXED : Tri Mode assert
--------------------------

::


    op --debugger --gltf 0  --tracer --rendermode +in0 --restrictmesh 5
          #
          #  asserts for lack of GParts in OGeo::makeAnalyticGeometry 
          #  ... this is from attempt to make both ana and tri 
          #  GPU geometries when ana not enabled 


    (lldb) bt
    * thread #1: tid = 0xac029, 0x00007fff8ae12866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff8ae12866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff824af35c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff891ffb1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff891c99bf libsystem_c.dylib`__assert_rtn + 321
        frame #4: 0x00000001035c1c1b libOptiXRap.dylib`OGeo::makeAnalyticGeometry(this=0x0000000115dcfd10, mm=0x000000010781b200) + 571 at OGeo.cc:609
        frame #5: 0x00000001035bfa7f libOptiXRap.dylib`OGeo::makeRepeatedGroup(this=0x0000000115dcfd10, mm=0x000000010781b200, lod=false) + 1439 at OGeo.cc:299
        frame #6: 0x00000001035bdf07 libOptiXRap.dylib`OGeo::convertMergedMesh(this=0x0000000115dcfd10, i=5) + 1671 at OGeo.cc:250
        frame #7: 0x00000001035bd2b4 libOptiXRap.dylib`OGeo::convert(this=0x0000000115dcfd10) + 340 at OGeo.cc:178
        frame #8: 0x00000001035b66e4 libOptiXRap.dylib`OScene::init(this=0x000000011261fc70) + 6196 at OScene.cc:163
        frame #9: 0x00000001035b4e4d libOptiXRap.dylib`OScene::OScene(this=0x000000011261fc70, hub=0x0000000105e00180) + 317 at OScene.cc:85
        frame #10: 0x00000001035b6c9d libOptiXRap.dylib`OScene::OScene(this=0x000000011261fc70, hub=0x0000000105e00180) + 29 at OScene.cc:87
        frame #11: 0x0000000103b4cd56 libOKOP.dylib`OpEngine::OpEngine(this=0x000000011261fc10, hub=0x0000000105e00180) + 182 at OpEngine.cc:43
        frame #12: 0x0000000103b4d21d libOKOP.dylib`OpEngine::OpEngine(this=0x000000011261fc10, hub=0x0000000105e00180) + 29 at OpEngine.cc:55
        frame #13: 0x0000000103c44a44 libOK.dylib`OKPropagator::OKPropagator(this=0x0000000112615d70, hub=0x0000000105e00180, idx=0x00000001095e2700, viz=0x00000001095e5b20) + 196 at OKPropagator.cc:44
        frame #14: 0x0000000103c44bbd libOK.dylib`OKPropagator::OKPropagator(this=0x0000000112615d70, hub=0x0000000105e00180, idx=0x00000001095e2700, viz=0x00000001095e5b20) + 45 at OKPropagator.cc:52
        frame #15: 0x0000000103c44377 libOK.dylib`OKMgr::OKMgr(this=0x00007fff5fbfe9f8, argc=9, argv=0x00007fff5fbfead8, argforced=0x000000010001580d) + 663 at OKMgr.cc:43
        frame #16: 0x0000000103c4464b libOK.dylib`OKMgr::OKMgr(this=0x00007fff5fbfe9f8, argc=9, argv=0x00007fff5fbfead8, argforced=0x000000010001580d) + 43 at OKMgr.cc:49
        frame #17: 0x000000010000a95d OTracerTest`main(argc=9, argv=0x00007fff5fbfead8) + 1373 at OTracerTest.cc:64
        frame #18: 0x00007fff862855fd libdyld.dylib`start + 1
        frame #19: 0x00007fff862855fd libdyld.dylib`start + 1
    (lldb) f 4
    frame #4: 0x00000001035c1c1b libOptiXRap.dylib`OGeo::makeAnalyticGeometry(this=0x0000000115dcfd10, mm=0x000000010781b200) + 571 at OGeo.cc:609
       606  
       607      // when using --test eg PmtInBox or BoxInBox the mesh is fabricated in GGeoTest
       608  
    -> 609      GParts* pts = mm->getParts(); assert(pts && "GMergedMesh with GEOCODE_ANALYTIC must have associated GParts, see GGeo::modifyGeometry "); 
       610  
       611      if(pts->getPrimBuffer() == NULL)




 

OptiX Geometry Selector (from 4.1.1 manual)
---------------------------------------------

A selector is similar to a group in that it is a collection of higher level
graph nodes. The number of nodes in the collection is set by
rtSelectorSetChildCount, and the individual children are assigned with
rtSelectorSetChild. Valid child types are rtGroup, rtGeometryGroup,
rtTransform, and rtSelector.

The main difference between selectors and groups is that selectors do not have
an acceleration structure associated with them. Instead, a visit program is
specified with rtSelectorSetVisitProgram. This program is executed every time a
ray encounters the selector node during graph traversal. The program specifies
which children the ray should continue traversal through by calling
rtIntersectChild.

A typical use case for a selector is dynamic (i.e. per-ray) level of detail: an
object in the scene may be represented by a number of geometry nodes, each
containing a different level of detail version of the object. The geometry
groups containing these different representations can be assigned as children
of a selector. The visit program can select which child to intersect using any
criterion (e.g. based on the footprint or length of the current ray), and
ignore the others.

As for groups and other graph nodes, child nodes of a selector can be shared
with other graph nodes to allow flexible instancing.


optixSelector sample
----------------------

Setup a pair of colocated sphere geometry groups with different radii, tie them together 
into a *Selector* with child count 2.

::

    226 void create_geometry( RTcontext context, RTmaterial material[] )
    227 {
    228     /* Setup two geometry groups */
    229 
    230     // Geometry nodes (two spheres at same position, but with different radii)
    231     RTgeometry geometry[2];
    232 
    233     geometry[0] = makeGeometry(context, 1);
    234     makeGeometryPrograms(context, geometry[0], "sphere.cu", "intersect", "bounds");
    235     makeGeometryVariable4f(context, geometry[0], "sphere", 0.0f, 0.0f, 0.0f, 0.5f);
    236 
    237     geometry[1] = makeGeometry(context, 1);
    238     makeGeometryPrograms(context, geometry[1], "sphere.cu", "intersect", "bounds");
    239     makeGeometryVariable4f(context, geometry[1], "sphere", 0.0f, 0.0f, 0.0f, 1.0f);
    240 
    241     // Geometry instance nodes
    242     RTgeometryinstance instance[2];
    243     instance[0] = makeGeometryInstance( context, geometry[0], material[0] );
    244     instance[1] = makeGeometryInstance( context, geometry[1], material[1] );
    245 
    246     // Accelerations nodes
    247     RTacceleration acceleration[2];
    248     acceleration[0] = makeAcceleration( context, "NoAccel" );
    249     acceleration[1] = makeAcceleration( context, "NoAccel" );
    250 
    251     // Geometry group nodes
    252     RTgeometrygroup group[2];
    253     group[0] = makeGeometryGroup( context, instance[0], acceleration[0] );
    254     group[1] = makeGeometryGroup( context, instance[1], acceleration[1] );
    255 
    256     /* Setup selector as top objects */
    257 
    258     // Init selector node
    259     RTselector selector;
    260     RTprogram  stor_visit_program;
    261     RT_CHECK_ERROR( rtSelectorCreate(context,&selector) );
    262     RT_CHECK_ERROR( rtProgramCreateFromPTXFile(context,ptxpath("selector_example.cu").c_str(),"visit",&stor_visit_program) );
    263     RT_CHECK_ERROR( rtSelectorSetVisitProgram(selector,stor_visit_program) );
    264     RT_CHECK_ERROR( rtSelectorSetChildCount(selector,2) );
    265     RT_CHECK_ERROR( rtSelectorSetChild(selector, 0, group[0]) );
    266     RT_CHECK_ERROR( rtSelectorSetChild(selector, 1, group[1]) );
    267 
    268     // Attach selector to context as top object
    269     RTvariable var_group;
    270     RT_CHECK_ERROR( rtContextDeclareVariable(context,"top_object",&var_group) );
    271     RT_CHECK_ERROR( rtVariableSetObject(var_group, selector) );
    272 }





Pick which geometry group to show based on ray direction::

     29 #include <optix.h>
     30 #include <optixu/optixu_math_namespace.h>
     31 
     32 using namespace optix;
     33 
     34 rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
     35 
     36 RT_PROGRAM void visit()
     37 {
     38   unsigned int index = (unsigned int)( ray.direction.y < 0.0f );
     39   rtIntersectChild( index );
     40 }



Hmm not clear how to structure 
----------------------------------


Looks like will need a separate selector for every instance... for instance identity::

      Selector
          GeometryGroup 
               GeometryInstance(Geometry,Material)
               Acceleration
          GeometryGroup 
               GeometryInstance(Geometry,Material)
               Acceleration


Rules
~~~~~~~

* Group contains : rtGroup, rtGeometryGroup, rtTransform, or rtSelector
* Transform houses single child : rtGroup, rtGeometryGroup, rtTransform, or rtSelector   (NB not GeometryInstance)
* GeometryGroup is a container for an arbitrary number of geometry instances, and must be assigned an Acceleration
* Selector contains : rtGroup, rtGeometryGroup, rtTransform, and rtSelector


Where to put Selector ? 
~~~~~~~~~~~~~~~~~~~~~~~~~~

Given that the same gmm is used for all pergi... 
it would seem most appropriate to arrange the selector in common also, 
as all instances have the same simplified version of their geometry too..
BUT: selector needs to house 


How to form a simplified analytic instance ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* used lazy LOD, just setting primCount to restrict to outermost volume for LOD level 1


How to select ?
~~~~~~~~~~~~~~~~~

Just like the OpenGL LOD : the level-of-detail decision needs access to: 

* instance position  (could get this using rtGetTransform, BUT tis known already in OGeo so set as visit program attribute)
* instance "size" 

When distance from ray.origin to instance (transform center) exceeds instance size
can select just the outer ?  


Program Variable Transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From the visit program, ray.origin appears to be in object space ? But instance position is in World space.

::

    visit_instance 1  ray.origin (   152.681    541.562    167.953)  instance_position (-17951.658 -795436.938  -3156.400      1.000)  
    visit_instance 1  ray.origin (   152.681    541.562    167.953)  instance_position (-17951.658 -795436.938  -3156.400      1.000)  
    visit_instance 1  ray.origin (   152.681    541.562    167.953)  instance_position (-17951.658 -795436.938  -3156.400      1.000)  
    visit_instance 1  ray.origin (   152.681    541.562    167.953)  instance_position (-17951.658 -795436.938  -3156.400      1.000)  
    visit_instance 1  ray.origin (   152.681    541.562    167.953)  instance_position (-17951.658 -795436.938  -3156.400      1.000)  
    visit_instance 1  ray.origin (   152.681    541.562    167.953)  instance_position (-17951.658 -795436.938  -3156.400      1.000)  
    visit_instance 1  ray.origin (   152.681    541.562    167.953)  instance_position (-17951.658 -795436.938  -3156.400      1.000)  
    visit_instance 1  ray.origin (   152.681    541.562    167.953)  instance_position (-17951.658 -795436.938  -3156.400      1.000)  
    visit_instance 1  ray.origin (   152.681    541.562    167.953)  instance_position (-17951.658 -795436.938  -3156.400      1.000)  
    visit_instance 1  ray.origin (   152.681    541.562    167.953)  instance_position (-17951.658 -795436.938  -3156.400      1.000)  


Recall that rays have a projective transformation applied to them upon encountering Transform nodes during traversal. 
The transformed ray is said to be in object space, while the original ray is said to be in world space.
Programs with access to the rtCurrentRay semantic operate in the spaces summarized in Table 7:

Table 7 Space of rtCurrentRay for Each Program Type

===============  =============
Program           Space
===============  =============
Ray Generation    World
Closest Hit       World
Any Hit           Object
Miss              World
Intersection      Object
Visit             Object
===============  =============

To facilitate transforming variables from one space to another, OptiX’s CUDA C API provides a set of functions::

   ￼__device__ float3 rtTransformPoint(  RTtransformkind kind, const float3& p )
    __device__ float3 rtTransformVector( RTtransformkind kind, const float3& v ) 
    __device__ float3 rtTransformNormal( RTtransformkind kind, const float3& n )
    __device__ void rtGetTransform( RTtransformkind kind, float matrix[16] )

The first three functions transform a float3, interpreted as a point, vector,
or normal vector, from object to world space or vice versa depending on the
value of a RTtransformkind flag passed as an argument. rtGetTransform returns
the four-by-four matrix representing the current transformation from object to
world space (or vice versa depending on the RTtransformkind argument). For best
performance, use the rtTransform functions rather than performing your own
explicit matrix multiplication with the result of rtGetTransform.

A common use case of variable transformation occurs when interpreting
attributes passed from the intersection program to the closest hit program.
Intersection programs often produce attributes, such as normal vectors, in
object space. Should a closest hit program wish to consume that attribute, it
often must transform the attribute from object space to world space:

::

    float3 n = rtTransformNormal( RT_OBJECT_TO_WORLD, normal );



After apply the transform get into ballpark::

    visit_instance 1  ray.origin (-20419.215 -799359.688  -6529.901)  instance_position (-17951.658 -795436.938  -3156.400      1.000)  
    visit_instance 1  ray.origin (-20419.215 -799359.688  -6529.901)  instance_position (-17951.658 -795436.938  -3156.400      1.000)  
    visit_instance 1  ray.origin (-20419.215 -799359.688  -6529.901)  instance_position (-17951.658 -795436.938  -3156.400      1.000)  
    visit_instance 1  ray.origin (-20419.215 -799359.688  -6529.901)  instance_position (-17951.658 -795436.938  -3156.400      1.000)  
    visit_instance 1  ray.origin (-20419.215 -799359.688  -6529.901)  instance_position (-17951.658 -795436.938  -3156.400      1.000)  
    visit_instance 1  ray.origin (-20419.215 -799359.688  -6529.901)  instance_position (-17951.658 -795436.938  -3156.400      1.000)  
    visit_instance 1  ray.origin (-20419.215 -799359.688  -6529.901)  instance_position (-17951.658 -795436.938  -3156.400      1.000)  
    visit_instance 1  ray.origin (-20419.215 -799359.688  -6529.901)  instance_position (-17951.658 -795436.938  -3156.400      1.000)  
    visit_instance 1  ray.origin (-20419.215 -799359.688  -6529.901)  instance_position (-17951.658 -795436.938  -3156.400      1.000)  





Attempt to test with selector between the analytic and triangulated geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* got slow OGeo convert, and GPU mem limit when tried making geo for each instance


* need to arrange a view with just instances to check the --raylod, 
  restrictmesh seems not to do it ?


::

    op --raylod --debugger --gltf 3  --tracer

    op --raylod --debugger --gltf 3  --tracer --restrictmesh 5




    op --debugger --gltf 3  --tracer --rendermode +in0,+in1,+in2,+in3,+in4,+in5

    op --debugger --gltf 3  --tracer --rendermode +in3
          # just PMTs in OpenGL, raytrace full geo (analytic)

    op --debugger --gltf 3  --tracer --rendermode +in3 --restrictmesh 3
          # OpenGL disappeared, raytrace still full geo (analytic)


FIXED : ana + tri ggeolib with inconsistent settings 
------------------------------------------------------

* regularize in OpticksHub::configureGeometryTriAna


::

    2017-10-17 15:44:53.647 INFO  [640131] [GGeoLib::dump@299] GGeoLib TRIANGULATED  numMergedMesh 6 ptr 0x105e3eca0
    mm i   0 geocode   K      SKIP        numSolids      12230 numFaces      403712 numITransforms           1 numITransforms*numSolids       12230
    mm i   1 geocode   K      SKIP  EMPTY numSolids          1 numFaces           0 numITransforms        1792 numITransforms*numSolids        1792
    mm i   2 geocode   K      SKIP        numSolids          1 numFaces          12 numITransforms         864 numITransforms*numSolids         864
    mm i   3 geocode   K      SKIP        numSolids          1 numFaces          12 numITransforms         864 numITransforms*numSolids         864
    mm i   4 geocode   K      SKIP        numSolids          1 numFaces          12 numITransforms         864 numITransforms*numSolids         864
    mm i   5 geocode   T                  numSolids          5 numFaces        2928 numITransforms         672 numITransforms*numSolids        3360
     num_total_volumes 12230 num_instanced_volumes 7744 num_global_volumes 4486


    2017-10-17 15:44:53.829 INFO  [640131] [GGeoLib::dump@299] GGeoLib ANALYTIC  numMergedMesh 6 ptr 0x108413550
    mm i   0 geocode   A                  numSolids      12230 numFaces      403712 numITransforms           1 numITransforms*numSolids       12230
    mm i   1 geocode   A            EMPTY numSolids          1 numFaces           0 numITransforms        1792 numITransforms*numSolids        1792
    mm i   2 geocode   A                  numSolids          1 numFaces          12 numITransforms         864 numITransforms*numSolids         864
    mm i   3 geocode   A                  numSolids          1 numFaces          12 numITransforms         864 numITransforms*numSolids         864
    mm i   4 geocode   A                  numSolids          1 numFaces          12 numITransforms         864 numITransforms*numSolids         864
    mm i   5 geocode   A                  numSolids          5 numFaces        2928 numITransforms         672 numITransforms*numSolids        3360
     num_total_volumes 12230 num_instanced_volumes 7744 num_global_volumes 4486



::


    248 void OpticksViz::uploadGeometry()
    249 {
    250     NPY<unsigned char>* colors = m_hub->getColorBuffer();
    251 
    252     m_scene->uploadColorBuffer( colors );  //     oglrap-/Colors preps texture, available to shaders as "uniform sampler1D Colors"
    253 
    254     LOG(info) << m_ok->description();
    255 
    256     m_composition->setTimeDomain(        m_ok->getTimeDomain() );
    257     m_composition->setDomainCenterExtent(m_ok->getSpaceDomain());
    258 
    259     m_scene->setGeometry(m_hub->getGeoLib());
    260 
    261     m_scene->uploadGeometry();
    262 
    263 
    264     m_hub->setupCompositionTargetting();
    265 
    266 }


    342 GGeoLib* OpticksHub::getGeoLib()
    343 {
    344     return m_ggeo->getGeoLib() ;
    345 }
    346 

    471 GGeoLib* GGeo::getGeoLib()
    472 {
    473     return m_geolib ;
    474 }



OScene::

    124     //m_ggeo = m_hub->getGGeo();
    125     m_ggeo = m_hub->getGGeoBase();
    126 
    127     LOG(info) << "OScene::init"
    128               << " ggeobase identifier : " << m_ggeo->getIdentifier()
    129               ;
    130 
    131 
    132     m_geolib = m_ggeo->getGeoLib();
    133 



gltf switch::

    350 GGeoBase* OpticksHub::getGGeoBase()
    351 {
    352    // analytic switch 
    353 
    354     GGeoBase* ggb = m_gltf ? dynamic_cast<GGeoBase*>(m_gscene) : dynamic_cast<GGeoBase*>(m_ggeo) ;
    355     LOG(info) << "OpticksHub::getGGeoBase"
    356               << " analytic switch  "
    357               << " m_gltf " << m_gltf
    358               << " ggb " << ( ggb ? ggb->getIdentifier() : "NULL" )
    359                ;
    360 
    361     return ggb ;
    362 }
    363 




getRestrictMesh
------------------


::

    339 void OpticksGeometry::configureGeometry()
    340 {
    341     int restrict_mesh = m_fcfg->getRestrictMesh() ;
    342     int analytic_mesh = m_fcfg->getAnalyticMesh() ;
    343 
    344     int nmm = m_ggeo->getNumMergedMesh();
    345 
    346     LOG(debug) << "OpticksGeometry::configureGeometry"
    347               << " restrict_mesh " << restrict_mesh
    348               << " analytic_mesh " << analytic_mesh
    349               << " nmm " << nmm
    350               ;
    351 
    352     std::string instance_slice = m_fcfg->getISlice() ;;
    353     std::string face_slice = m_fcfg->getFSlice() ;;
    354     std::string part_slice = m_fcfg->getPSlice() ;;
    355 
    356     NSlice* islice = !instance_slice.empty() ? new NSlice(instance_slice.c_str()) : NULL ;
    357     NSlice* fslice = !face_slice.empty() ? new NSlice(face_slice.c_str()) : NULL ;
    358     NSlice* pslice = !part_slice.empty() ? new NSlice(part_slice.c_str()) : NULL ;
    359 
    360     for(int i=0 ; i < nmm ; i++)
    361     {
    362         GMergedMesh* mm = m_ggeo->getMergedMesh(i);
    363         if(!mm) continue ;
    364 
    365         if(restrict_mesh > -1 && i != restrict_mesh ) mm->setGeoCode(OpticksConst::GEOCODE_SKIP);
    366         if(analytic_mesh > -1 && i == analytic_mesh && i > 0)
    367         {
    368             GPmt* pmt = m_ggeo->getPmt();
    369             assert(pmt && "analyticmesh requires PMT resource");
    370 
    371             GParts* analytic = pmt->getParts() ;
    372             // TODO: the strings should come from config, as detector specific
    373 
    374             analytic->setVerbosity(m_verbosity);
    375             analytic->setContainingMaterial("MineralOil");
    376             analytic->setSensorSurface("lvPmtHemiCathodeSensorSurface");
    377 
    378             mm->setGeoCode(OpticksConst::GEOCODE_ANALYTIC);
    379             mm->setParts(analytic);
    380         }
    381         if(i>0) mm->setInstanceSlice(islice);
    382 
    383         // restrict to non-global for now
    384         if(i>0) mm->setFaceSlice(fslice);
    385         if(i>0) mm->setPartSlice(pslice);
    386     }
    387 
    388     TIMER("configureGeometry");
    389 }





Renderer and mesh indices not aligned ?   
-----------------------------------------------

* inconsistent criteria ? 

  * no, seems just the renderer has dynamic index, not tied to index of geometry 

* TODO: get the name of the instanced mesh into the interface, or at least dump it 


::

    2017-10-16 19:27:59.433 INFO  [511521] [GGeoLib::dump@298] GGeoLib ANALYTIC  numMergedMesh 6
    mm i   0 geocode   A                  numSolids      12230 numFaces      403712 numITransforms           1 numITransforms*numSolids       12230
    mm i   1 geocode   A            EMPTY numSolids          1 numFaces           0 numITransforms        1792 numITransforms*numSolids        1792
    mm i   2 geocode   A                  numSolids          1 numFaces          12 numITransforms         864 numITransforms*numSolids         864
    mm i   3 geocode   A                  numSolids          1 numFaces          12 numITransforms         864 numITransforms*numSolids         864
    mm i   4 geocode   A                  numSolids          1 numFaces          12 numITransforms         864 numITransforms*numSolids         864
    mm i   5 geocode   A                  numSolids          5 numFaces        2928 numITransforms         672 numITransforms*numSolids        3360
     num_total_volumes 12230 num_instanced_volumes 7744 num_global_volumes 4486
    2017-10-16 19:27:59.433 WARN  [511521] [OGeo::convertMergedMesh@224]  RayLOD enabled 
    2017-10-16 19:27:59.656 WARN  [511521] [OGeo::convertMergedMesh@224]  RayLOD enabled 
    2017-10-16 19:27:59.656 WARN  [511521] [OGeo::convertMergedMesh@229] OGeo::convertMesh skipping mesh 1
    2017-10-16 19:27:59.656 WARN  [511521] [OGeo::convertMergedMesh@224]  RayLOD enabled 
    2017-10-16 19:27:59.660 FATAL [511521] [*GMesh::makeFaceRepeatedInstancedIdentityBuffer@1997] GMesh::makeFaceRepeatedInstancedIdentityBuffer nodeinfo_ok 1 nodeinfo_buffer_items 1 numSolids 1
    2017-10-16 19:27:59.660 FATAL [511521] [*GMesh::makeFaceRepeatedInstancedIdentityBuffer@2005] GMesh::makeFaceRepeatedInstancedIdentityBuffer iidentity_ok 1 iidentity_buffer_items 864 numFaces (sum of faces in numSolids)12 numITransforms 864 numSolids*numITransforms 864 numRepeatedIdentity 10368
    [ 2] (     0/   864 )         ip-20119.562 -796322.625 -9913.898   1.000 
    [ 2] (     1/   864 )         ip-20253.062 -796409.000 -9822.100   1.000 
    [ 2] (     2/   864 )         ip-20119.562 -796322.625 -9730.301   1.000 
    [ 2] (     3/   864 )         ip-19251.227 -795760.875 -9766.898   1.000 
    [ 2] (     4/   864 )         ip-19384.727 -795847.250 -9675.100   1.000 
    [ 2] (     5/   864 )         ip-19251.227 -795760.875 -9583.301   1.000 
    [ 2] (     6/   864 )         ip-21102.336 -796958.375 -9766.898   1.000 
    [ 2] (     7/   864 )         ip-21235.836 -797044.750 -9675.100   1.000 
    [ 2] (     8/   864 )         ip-21102.336 -796958.375 -9583.301   1.000 
    [ 2] (     9/   864 )         ip-20119.398 -796322.875 -7676.898   1.000 
    2017-10-16 19:27:59.756 WARN  [511521] [OGeo::convertMergedMesh@224]  RayLOD enabled 
    2017-10-16 19:27:59.756 FATAL [511521] [*GMesh::makeFaceRepeatedInstancedIdentityBuffer@1997] GMesh::makeFaceRepeatedInstancedIdentityBuffer nodeinfo_ok 1 nodeinfo_buffer_items 1 numSolids 1
    2017-10-16 19:27:59.756 FATAL [511521] [*GMesh::makeFaceRepeatedInstancedIdentityBuffer@2005] GMesh::makeFaceRepeatedInstancedIdentityBuffer iidentity_ok 1 iidentity_buffer_items 864 numFaces (sum of faces in numSolids)12 numITransforms 864 numSolids*numITransforms 864 numRepeatedIdentity 10368
    [ 3] (     0/   864 )         ip-20079.611 -796362.250 -9934.684   1.000 
    [ 3] (     1/   864 )         ip-20243.338 -796468.188 -9822.100   1.000 
    [ 3] (     2/   864 )         ip-20079.611 -796362.250 -9709.517   1.000 
    [ 3] (     3/   864 )         ip-19211.277 -795800.500 -9787.684   1.000 
    [ 3] (     4/   864 )         ip-19375.004 -795906.438 -9675.100   1.000 
    [ 3] (     5/   864 )         ip-19211.277 -795800.500 -9562.517   1.000 
    [ 3] (     6/   864 )         ip-21062.387 -796998.062 -9787.684   1.000 
    [ 3] (     7/   864 )         ip-21226.113 -797104.000 -9675.100   1.000 
    [ 3] (     8/   864 )         ip-21062.387 -796998.062 -9562.517   1.000 
    [ 3] (     9/   864 )         ip-20079.449 -796362.500 -7697.684   1.000 
    2017-10-16 19:27:59.790 WARN  [511521] [OGeo::convertMergedMesh@224]  RayLOD enabled 
    2017-10-16 19:27:59.790 FATAL [511521] [*GMesh::makeFaceRepeatedInstancedIdentityBuffer@1997] GMesh::makeFaceRepeatedInstancedIdentityBuffer nodeinfo_ok 1 nodeinfo_buffer_items 1 numSolids 1
    2017-10-16 19:27:59.790 FATAL [511521] [*GMesh::makeFaceRepeatedInstancedIdentityBuffer@2005] GMesh::makeFaceRepeatedInstancedIdentityBuffer iidentity_ok 1 iidentity_buffer_items 864 numFaces (sum of faces in numSolids)12 numITransforms 864 numSolids*numITransforms 864 numRepeatedIdentity 10368
    [ 4] (     0/   864 )         ip-20066.975 -796431.500 -9887.918   1.000 
    [ 4] (     1/   864 )         ip-20162.691 -796493.438 -9822.100   1.000 
    [ 4] (     2/   864 )         ip-20066.975 -796431.500 -9756.282   1.000 
    [ 4] (     3/   864 )         ip-19198.641 -795869.750 -9740.918   1.000 
    [ 4] (     4/   864 )         ip-19294.357 -795931.688 -9675.100   1.000 
    [ 4] (     5/   864 )         ip-19198.641 -795869.750 -9609.282   1.000 
    [ 4] (     6/   864 )         ip-21049.750 -797067.312 -9740.918   1.000 
    [ 4] (     7/   864 )         ip-21145.467 -797129.188 -9675.100   1.000 
    [ 4] (     8/   864 )         ip-21049.750 -797067.312 -9609.282   1.000 
    [ 4] (     9/   864 )         ip-20066.812 -796431.750 -7650.918   1.000 
    2017-10-16 19:27:59.823 WARN  [511521] [OGeo::convertMergedMesh@224]  RayLOD enabled 
    2017-10-16 19:27:59.823 FATAL [511521] [*GMesh::makeFaceRepeatedInstancedIdentityBuffer@1997] GMesh::makeFaceRepeatedInstancedIdentityBuffer nodeinfo_ok 1 nodeinfo_buffer_items 5 numSolids 5
    2017-10-16 19:27:59.823 FATAL [511521] [*GMesh::makeFaceRepeatedInstancedIdentityBuffer@2005] GMesh::makeFaceRepeatedInstancedIdentityBuffer iidentity_ok 1 iidentity_buffer_items 3360 numFaces (sum of faces in numSolids)2928 numITransforms 672 numSolids*numITransforms 3360 numRepeatedIdentity 1967616
    [ 5] (     0/   672 )         ip-16572.898 -801469.625 -8842.500   1.000 
    [ 5] (     1/   672 )         ip-16166.072 -801019.375 -8842.500   1.000 
    [ 5] (     2/   672 )         ip-15889.641 -800479.188 -8842.500   1.000 
    [ 5] (     3/   672 )         ip-15762.440 -799885.875 -8842.500   1.000 
    [ 5] (     4/   672 )         ip-15793.142 -799279.812 -8842.500   1.000 
    [ 5] (     5/   672 )         ip-15979.650 -798702.375 -8842.500   1.000 
    [ 5] (     6/   672 )         ip-16309.258 -798192.875 -8842.500   1.000 
    [ 5] (     7/   672 )         ip-16759.500 -797786.062 -8842.500   1.000 
    [ 5] (     8/   672 )         ip-17299.695 -797509.625 -8842.500   1.000 
    [ 5] (     9/   672 )         ip-17893.031 -797382.438 -8842.500   1.000 








