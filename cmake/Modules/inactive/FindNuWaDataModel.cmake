#  Depends on envvar LOCAL_BASE which is set via env- precursor bash function

set(DATAMODEL_PREFIX "$ENV{LOCAL_BASE}/env/nuwa")
set(DATAMODEL_LIBRARIES "${DATAMODEL_PREFIX}/lib/libDataModel.dylib")
set(DATAMODEL_INCLUDE_DIRS "${DATAMODEL_PREFIX}/include")
set(DATAMODEL_DEFINITIONS "-DGOD_NOALLOC")
