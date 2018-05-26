
set(OpenVR_PREFIX "$ENV{LOCAL_BASE}/env/vr/openvr")

find_library( OpenVR_LIBRARIES 
              NAMES openvr_api
              PATHS ${OpenVR_PREFIX}/lib/osx32 )

set(OpenVR_INCLUDE_DIRS "${OpenVR_PREFIX}/headers")
set(OpenVR_DEFINITIONS "")

