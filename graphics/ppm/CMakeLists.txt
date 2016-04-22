cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name PPM)
project(${name})

add_executable(${name}Test main.cc)

SET(hdrs 
      loadPPM.h
)

install(TARGETS ${name}Test DESTINATION bin)
install(FILES ${hdrs} DESTINATION include/${name})
