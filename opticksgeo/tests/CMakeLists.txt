cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name OpticksGeometryTest)
project(${name})

#file(GLOB TEST_CC_SRCS "*Test.cc")

set(TEST_SOURCES
    OpticksGeometryTest.cc
)



foreach(TEST_CC_SRC ${TEST_SOURCES})
    get_filename_component(TGT ${TEST_CC_SRC} NAME_WE)
    add_executable(${TGT} ${TEST_CC_SRC})

    add_test(${name}.${TGT} ${TGT})
    add_dependencies(check ${TGT})

    target_link_libraries(${TGT} 
               ${LIBRARIES} OpticksGeometry
    )
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()


# cmakex-
#set_property(TEST ${name}.OpticksResourceTest PROPERTY ENVIRONMENT 
#   "OPTICKS_GEOKEY=DAE_NAME_SPANNER"
#   "DAE_NAME_SPANNER=/tmp/spanner_in_works.dae"
# )

