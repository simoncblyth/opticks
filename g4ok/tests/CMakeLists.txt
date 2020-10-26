cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name G4OKTest)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

set(TESTS
    G4OKTest.cc
    MockSensorAngularEfficiencyTableTest.cc
)

foreach(SRC ${TESTS})

    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    target_link_libraries(${TGT} G4OK)
    install(TARGETS ${TGT} DESTINATION lib)

    add_test(${name}.${TGT} ${TGT})
    #add_dependencies(check ${TGT})

#    add_test(${name}.${TGT}_g4gun ${TGT} --g4gun)   ## add test with argument : --g4gun

endforeach()





