cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
set(name PPM)
project(${name})

add_executable(${name}Test main.cc)
install(TARGETS ${name}Test DESTINATION bin)

SET(hdrs 
      loadPPM.h
)

install(FILES ${hdrs} DESTINATION include)
install(FILES ${hdrs} DESTINATION include)
