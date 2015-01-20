#
#  Depends on envvar LOCAL_BASE which is set via env- precursor bash function
#

find_path(
    ZMQROOT_INCLUDE_DIR 
    NAMES ZMQRoot.hh
    PATHS "$ENV{LOCAL_BASE}/env/zmqroot"
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
)

find_library(
    ZMQROOT_LIBRARY
    NAMES ZMQRoot
    PATHS "$ENV{LOCAL_BASE}/env/zmqroot"
    PATH_SUFFIXES lib
    NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    ZMQRoot
    DEFAULT_MSG
    ZMQROOT_LIBRARY
    ZMQROOT_INCLUDE_DIR
)

if (ZMQROOT_FOUND)
    set(ZMQROOT_LIBRARIES ${ZMQROOT_LIBRARY})
    set(ZMQROOT_INCLUDE_DIRS ${ZMQROOT_INCLUDE_DIR})
else (ZMQROOT_FOUND)
    set(ZMQROOT_LIBRARIES)
    set(ZMQROOT_INCLUDE_DIRS)
endif (ZMQROOT_FOUND)

mark_as_advanced(
    ZMQROOT_LIBRARY 
    ZMQROOT_INCLUDE_DIR
)

