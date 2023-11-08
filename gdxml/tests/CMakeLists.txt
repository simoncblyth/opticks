cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name GDXMLTest)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)   

set(INSTALLED_SCRIPTS
    GDTestRunner.sh 
)
install(PROGRAMS ${INSTALLED_SCRIPTS} DESTINATION bin)


set(TEST_SOURCES
   GDXMLTest.cc
)


#find_program(BASH_EXECUTABLE NAMES bash REQUIRED)
#message(STATUS "BASH_EXECUTABLE : ${BASH_EXECUTABLE}")

foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    target_link_libraries(${TGT} GDXML)
    install(TARGETS ${TGT} DESTINATION lib)

    #add_test(${name}.${TGT} ${TGT})
    add_test(
       NAME ${name}.${TGT} 
       COMMAND bash GDTestRunner.sh ${TGT}
       #COMMAND ${BASH_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/GDTestRunner.sh ${TGT}
    )
endforeach()


