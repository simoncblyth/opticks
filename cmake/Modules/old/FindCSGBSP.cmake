
set(CSGBSP_PREFIX "${OPTICKS_PREFIX}/externals")
set(CSGBSP_CPP "${CSGBSP_PREFIX}/csgbsp/csgjs-cpp/csgjs.cpp")


if(EXISTS ${CSGBSP_CPP})
    set(CSGBSP_FOUND TRUE)
    set(CSGBSP_DEFINITIONS "")
    set(CSGBSP_LIBRARIES "")
    set(CSGBSP_INCLUDE_DIRS "${CSGBSP_PREFIX}/csgbsp/csgjs-cpp")
else()
    set(CSGBSP_FOUND FALSE)
endif()


