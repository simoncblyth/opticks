cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name OKConfTest)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)


set(TEST_SOURCES
    OKConfTest.cc
)


if(TARGET OKConf)
   message( STATUS  "target OKConf exists")
endif()
if(TARGET Opticks::OKConf)
   message( STATUS "target Opticks::OKConf exists")
endif()


foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TGT ${TEST_SOURCE} NAME_WE)
    add_executable(${TGT} ${TEST_SOURCE})

    set(testname ${name}.${TGT})  

    # see cmakex-
    add_test(${testname} ${TGT})

    # this works proj-by-proj but not integrated 
    target_link_libraries(${TGT} PRIVATE OKConf) 

    # with integrated the lib is the installed one, which is wrong  


    #target_link_libraries(${TGT} PUBLIC OKConf) 
    # this is needed for the integrated build as
    # dont have installed headers to pull in 
    #target_include_directories( ${TGT} PUBLIC 
    #   $<BUILD_INTERFACE:${OKConf_SOURCE_DIR}>
    #   $<BUILD_INTERFACE:${OKConf_BINARY_DIR}/inc>
    #)

    target_include_directories( ${TGT} PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../inc>
    )         


    install(TARGETS ${TGT} DESTINATION lib)
endforeach()



