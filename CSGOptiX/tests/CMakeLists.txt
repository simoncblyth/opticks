cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name CSGOptiXTest)
project(${name} VERSION 0.1.0)


set(SINGLE_SOURCES
    CSGOptiXVersion.cc
    CSGOptiXRender.cc
    CSGOptiXSimulate.cc
)

foreach(SRC ${SINGLE_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC})
    target_link_libraries(${TGT} CSGOptiX)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()


