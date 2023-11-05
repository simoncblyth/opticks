
set(BoostAsio_MODULE ${CMAKE_CURRENT_LIST_FILE})

if(BoostAsio_VERBOSE)
message(STATUS "BoostAsio_MODULE : ${BoostAsio_MODULE}" )
endif()

find_path(
    BoostAsio_INCLUDE_DIR 
    NAMES "boost/asio.hpp"
    #NAMES "boost/asio.hpp.MAKE_IT_FAIL"
)

if(BoostAsio_INCLUDE_DIR)
   set(BoostAsio_FOUND "YES")
else()
   set(BoostAsio_FOUND "NO")
endif()

if(BoostAsio_VERBOSE)
    message(STATUS "FindBoostAsio.cmake : BoostAsio_MODULE      : ${BoostAsio_MODULE} ")
    message(STATUS "FindBoostAsio.cmake : BoostAsio_INCLUDE_DIR : ${BoostAsio_INCLUDE_DIR} ")
    message(STATUS "FindBoostAsio.cmake : BoostAsio_FOUND       : ${BoostAsio_FOUND}  ")
endif()


