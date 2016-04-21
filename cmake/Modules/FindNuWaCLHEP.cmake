#  Depends on envvar LOCAL_BASE which is set via env- precursor bash function

set(CLHEP_LIBRARIES "${CLHEP_PREFIX}/lib/libCLHEP.so")
set(CLHEP_INCLUDE_DIRS "$ENV{DYB}/external/clhep/2.0.4.2/x86_64-slc6-gcc44-dbg/include")

