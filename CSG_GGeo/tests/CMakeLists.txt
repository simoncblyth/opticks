cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name CSG_GGeoTest)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)   

set(TEST_SOURCES
   CSG_GGeoTest.cc 
)




#add_executable(${name}Test ${name}Test.cc)
#target_link_libraries(${name}Test PUBLIC Opticks::GGeo Opticks::CSG ${name})



foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})

    add_test(${name}.${TGT} ${TGT})
    #add_dependencies(check ${TGT})

    target_link_libraries(${TGT} CSG_GGeo)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()




