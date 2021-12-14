cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name CSG_GGeoTest)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)   

set(TEST_SOURCES
   CSG_GGeoTest.cc 
)


foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})

    #add_test(${name}.${TGT} ${TGT})   ## prevent from running as standard with tests

    target_link_libraries(${TGT} CSG_GGeo)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()




