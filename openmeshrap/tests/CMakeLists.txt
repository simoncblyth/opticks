cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name OpenMeshRapTest)
project(${name} VERSION 0.1.0)

#file(GLOB TEST_SOURCES "*Test.cc")
set(TEST_SOURCES

    DeleteFaceTest.cc
)

foreach(TEST_CC_SRC ${TEST_SOURCES})
    get_filename_component(TGT ${TEST_CC_SRC} NAME_WE)
    add_executable(${TGT} ${TEST_CC_SRC})

    add_test(${name}.${TGT} ${TGT})
    #add_dependencies(check ${TGT})

    target_link_libraries(${TGT} ${LIBRARIES} OpenMeshRap)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()

