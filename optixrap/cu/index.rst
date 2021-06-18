oxrap/cu index
==================

Overview
----------

Note there are two tracks that are mostly separate, 
but share geometry intersection. 

* propagation of photons
* radiance image creation 

TODO
-----

* major source cleanup
* renaming and organization into folders whilst retaining 
  the general structure of very few .cu and lots of .h which nvcc likes 


* changing .cu requires CMakeLists.txt changes, but .h does not : although sometimes need to touch CMakeLists.txt


.cu
-----

::

    ## top level ray generation RT_PROGRAM

    generate.cu
    pinhole_camera.cu

    ## closest hit RT_PROGRAM

    material1_propagate.cu
    material1_radiance.cu


* :doc:`material1_propagate.cu`



.h
-----

::

    ## general basis 

    switches.h
    quad.h        unions

    ## math 

    random.h
    rotateUz.h

    ## intersection RT_PROGRAM

    TriangleMesh.cu          (TODO: rename ~ "intersect_triangulated.cu" )
    intersect_analytic.cu    (formerly hemi-pmt.cu)

    ## intersection headers

    hemi-pmt.h            cylinder branch enum (was used by pinhole_camera.cu for visual debug ?) 
    bbox.h                transform bbox and transform testing 

    # original z-partlist approach 

    intersect_prism.h
    intersect_box.h
    intersect_zsphere.h
    intersect_ztubs.h

    # csg node tree approach 

    csg_intersect_boolean.h   evaluative_csg (node approach), intersect_csg (bileaf approach), UNSUPPORTED_recursive_csg
    csg_intersect_part.h      csg_intersect_part, csg_intersect_box, csg_intersect_sphere 
                              (used by csg_intersect_boolean)


    ## intersection "non-live" code

    boolean_solid.h       boolean combination enum tables, now dead as using compressed tables ?
                          (not dead but not live either, used to generate boolean_solid.py which is 
                           how the compressed tables are created)


    ## propagation basis

    PerRayData_propagate.h
    enums.h       TODO: elimate as duplicitous, get enumerations from CPU side header __CUDA__ sections
    state.h       propagation state filling 
    photon.h      persisting photons and records

    ## step loading and generation 

    torchstep.h
    cerenkovstep.h        
    scintillationstep.h  

    ## texture accessors

    reemission_lookup.h
    source_lookup.h
    boundary_lookup.h

    ## optical simulation 
    
    propagate.h    propagate_to_boundary, ...
    rayleigh.h     scattering 





    ## camera basis, helpers

    PerRayData_radiance.h
    color_lookup.h

    ## dead? code 

    helpers.h
    boolean-solid-dev.h
    intersect_part_dev.h




tests .cu
------------

::

    sphere.cu
    constantbg.cu

    LTminimalTest.cu
    OEventTest.cu
    OInterpolationTest.cu
    ORayleighTest.cu
    seedTest.cu
    axisTest.cu
    boundaryTest.cu
    boundaryLookupTest.cu
    compactionTest.cu
    dirtyBufferTest.cu
    bufferTest.cu
    minimalTest.cu
    tex0Test.cu
    texTest.cu
    textureTest.cu


