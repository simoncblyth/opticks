cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name GeoChainTest)
project(${name} VERSION 0.1.0)

set(TEST_SOURCES
    GeoChainSolidTest.cc
    GeoChainVolumeTest.cc
)

foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})

    add_test(${name}.${TGT} ${TGT})

    target_link_libraries(${TGT} Opticks::GeoChain)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()

