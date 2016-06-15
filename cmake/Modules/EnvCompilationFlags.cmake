


if(WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W4")

else(WIN32)

  ## c++11 forced by AsioZMQ : AsioZMQ not used here, but expect best to use same compiler options as far as possible
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++0x")  ## huh nvcc compilation fails with this ???
  if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -stdlib=libc++")
  else ()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++0x")
      #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")   #needed for numpyserver- on Linux ?
  endif ()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")
  if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-comment")
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")
  else()
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-private-field")
  endif()

endif(WIN32)


#message(" CMAKE_CXX_FLAGS : ${CMAKE_CXX_FLAGS} " )


