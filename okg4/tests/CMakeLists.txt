cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name okg4Test)
project(${name})

set(TESTS
    OKG4_LOGTest.cc
)

foreach(SRC ${TESTS})

    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    target_link_libraries(${TGT} ${LIBRARIES} okg4)
    install(TARGETS ${TGT} DESTINATION lib)

    add_test(${name}.${TGT} ${TGT})
    add_dependencies(check ${TGT})

endforeach()

