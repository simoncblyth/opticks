runtime_angular_and_way_control
=================================


Objective 
------------

Add runtime control inside the inconvenient on their own compiletime switches.

WITH_ANGULAR
   angular efficiency culling 
   --angular Opticks::isAngularEnabled

WITH_WAY_BUFFER
   way point recording 
   --way Opticks::isWayEnabled



angular 
---------------------------------------

The major difference with angular is that the PRD changes, handle
that at runtime by splitting into two materials::

    epsilon:cu blyth$ git mv material1_propagate.cu closest_hit_propagate.cu
    epsilon:cu blyth$ mv material1_angular_propagate.cu closest_hit_angular_propagate.cu



avoid quadupling sources with oxrap/cu/preprocessor.py which generates as specified by flags 
---------------------------------------------------------------------------------------------

oxrap/CMakeLists.txt::

    set(flags_AW +WITH_ANGULAR,+WITH_WAY_BUFFER)
    set(flags_Aw +WITH_ANGULAR,-WITH_WAY_BUFFER)
    set(flags_aW -WITH_ANGULAR,+WITH_WAY_BUFFER)
    set(flags_aw -WITH_ANGULAR,-WITH_WAY_BUFFER)


    add_custom_command(
        OUTPUT  generate_${flags_AW}.cu
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cu/preprocessor.py ${CMAKE_CURRENT_SOURCE_DIR}/cu/generate.cu --flags="${flags_AW}" --out ${CMAKE_CURRENT_BINARY_DIR}/generate_${flags_AW}.cu
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cu/generate.cu
    )
    add_custom_command(
        OUTPUT  generate_${flags_Aw}.cu
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cu/preprocessor.py ${CMAKE_CURRENT_SOURCE_DIR}/cu/generate.cu --flags="${flags_Aw}" --out ${CMAKE_CURRENT_BINARY_DIR}/generate_${flags_Aw}.cu
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cu/generate.cu
    )
    add_custom_command(
        OUTPUT  generate_${flags_aW}.cu
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cu/preprocessor.py ${CMAKE_CURRENT_SOURCE_DIR}/cu/generate.cu --flags="${flags_aW}" --out ${CMAKE_CURRENT_BINARY_DIR}/generate_${flags_aW}.cu
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cu/generate.cu
    )
    add_custom_command(
        OUTPUT  generate_${flags_aw}.cu
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cu/preprocessor.py ${CMAKE_CURRENT_SOURCE_DIR}/cu/generate.cu --flags="${flags_aw}" --out ${CMAKE_CURRENT_BINARY_DIR}/generate_${flags_aw}.cu
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cu/generate.cu
    )


checking are using the generated cu and ptx
----------------------------------------------

::

    epsilon:optixrap blyth$ l /usr/local/opticks/installcache/PTX/OptiXRap_generated_generate_*
    -rw-r--r--  1 blyth  staff  480674 Feb 16 22:21 /usr/local/opticks/installcache/PTX/OptiXRap_generated_generate_+WITH_ANGULAR,-WITH_WAY_BUFFER.cu.ptx
    -rw-r--r--  1 blyth  staff  480348 Feb 16 22:21 /usr/local/opticks/installcache/PTX/OptiXRap_generated_generate_-WITH_ANGULAR,+WITH_WAY_BUFFER.cu.ptx
    -rw-r--r--  1 blyth  staff  482497 Feb 16 22:21 /usr/local/opticks/installcache/PTX/OptiXRap_generated_generate_+WITH_ANGULAR,+WITH_WAY_BUFFER.cu.ptx
    -rw-r--r--  1 blyth  staff  478825 Feb 16 22:21 /usr/local/opticks/installcache/PTX/OptiXRap_generated_generate_-WITH_ANGULAR,-WITH_WAY_BUFFER.cu.ptx

::

    epsilon:opticks blyth$ OContext=INFO OConfig=INFO OKTest 


    2021-02-16 22:51:17.450 INFO  [11247997] [OpSeeder::seedComputeSeedsFromInteropGensteps@82] OpSeeder::seedComputeSeedsFromInteropGensteps : WITH_SEED_BUFFER 
    2021-02-16 22:51:17.480 INFO  [11247997] [OConfig::createProgram@114]  cu_name generate_-WITH_ANGULAR,-WITH_WAY_BUFFER.cu progname generate m_cmake_target OptiXRap m_ptxrel (null) path /usr/local/opticks/installcache/PTX/OptiXRap_generated_generate_-WITH_ANGULAR,-WITH_WAY_BUFFER.cu.ptx
    2021-02-16 22:51:17.651 INFO  [11247997] [OConfig::createProgram@114]  cu_name generate_-WITH_ANGULAR,-WITH_WAY_BUFFER.cu progname exception m_cmake_target OptiXRap m_ptxrel (null) path /usr/local/opticks/installcache/PTX/OptiXRap_generated_generate_-WITH_ANGULAR,-WITH_WAY_BUFFER.cu.ptx
    2021-02-16 22:51:17.659 INFO  [11247997] [OConfig::createProgram@114]  cu_name pinhole_camera.cu progname pinhole_camera m_cmake_target OptiXRap m_ptxrel (null) path /usr/local/opticks/installcache/PTX/OptiXRap_generated_pinhole_camera.cu.ptx
    2021-02-16 22:51:17.777 INFO  [11247997] [OConfig::createProgram@114]  cu_name pinhole_camera.cu progname exception m_cmake_target OptiXRap m_ptxrel (null) path /usr/local/opticks/installcache/PTX/OptiXRap_generated_pinhole_camera.cu.ptx
    2021-02-16 22:51:17.781 INFO  [11247997] [OConfig::createProgram@114]  cu_name constantbg.cu progname miss m_cmake_target OptiXRap m_ptxrel (null) path /usr/local/opticks/installcache/PTX/OptiXRap_generated_constantbg.cu.ptx



    epsilon:opticks blyth$ OContext=INFO OConfig=INFO OKTest --way --angular --compute

    2021-02-16 22:53:41.453 ERROR [11250092] [SensorLib::close@362]  SKIP as m_sensor_num zero 
    2021-02-16 22:53:41.479 FATAL [11250092] [*OCtx::create_buffer@300] skip upload_buffer as num_bytes zero key:OSensorLib_sensor_data
    2021-02-16 22:53:41.505 FATAL [11250092] [*OCtx::create_buffer@300] skip upload_buffer as num_bytes zero key:OSensorLib_texid
    2021-02-16 22:53:41.541 INFO  [11250092] [OConfig::createProgram@114]  cu_name generate_+WITH_ANGULAR,+WITH_WAY_BUFFER.cu progname generate m_cmake_target OptiXRap m_ptxrel (null) path /usr/local/opticks/installcache/PTX/OptiXRap_generated_generate_+WITH_ANGULAR,+WITH_WAY_BUFFER.cu.ptx
    2021-02-16 22:53:41.729 INFO  [11250092] [OConfig::createProgram@114]  cu_name generate_+WITH_ANGULAR,+WITH_WAY_BUFFER.cu progname exception m_cmake_target OptiXRap m_ptxrel (null) path /usr/local/opticks/installcache/PTX/OptiXRap_generated_generate_+WITH_ANGULAR,+WITH_WAY_BUFFER.cu.ptx
    2021-02-16 22:53:41.737 INFO  [11250092] [OContext::launch@810]  entry 0 width 0 height 0   printLaunchIndex ( -1 -1 -1) -
    2021-02-16 22:53:41.740 INFO  [11250092] [OContext::launch@823] VALIDATE time: 0.003286



