# NOT currently active
set(ZMQ_PREFIX "${OPTICKS_PREFIX}/externals")

find_path(
    ZMQ_INCLUDE_DIR 
    NAMES zmq.h
    PATHS "$ZMQ_PREFIX/zeromq"
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
)

find_library(
    ZMQ_LIBRARY
    NAMES zmq
    PATHS "$ZMQ_PREFIX/zeromq"
    PATH_SUFFIXES lib
    NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    ZMQ 
    DEFAULT_MSG
    ZMQ_LIBRARY
    ZMQ_INCLUDE_DIR
)

if (ZMQ_FOUND)
    set(ZMQ_LIBRARIES ${ZMQ_LIBRARY})
    set(ZMQ_INCLUDE_DIRS ${ZMQ_INCLUDE_DIR})
else (ZMQ_FOUND)
    set(ZMQ_LIBRARIES)
    set(ZMQ_INCLUDE_DIRS)
endif (ZMQ_FOUND)

mark_as_advanced(
    ZMQ_LIBRARY 
    ZMQ_INCLUDE_DIR
)

