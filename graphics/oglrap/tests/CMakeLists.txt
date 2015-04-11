cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name OGLRapTest)
project(${name})

add_executable( ${name} ${name}.cc )
add_executable( TrackballTest TrackballTest.cc )

target_link_libraries( ${name} ${LIBRARIES} OGLRap )
target_link_libraries( TrackballTest ${LIBRARIES} OGLRap )

install(TARGETS ${name} DESTINATION bin)
install(TARGETS TrackballTest DESTINATION bin)




