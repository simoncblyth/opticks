#  Depends on envvar LOCAL_BASE which is set via env- precursor bash function

set(G4DAECHROMA_PREFIX "$ENV{LOCAL_BASE}/env/chroma/G4DAEChroma")
set(G4DAECHROMA_LIBRARIES "${G4DAECHROMA_PREFIX}/lib/libG4DAEChroma.dylib")
set(G4DAECHROMA_INCLUDE_DIRS "${G4DAECHROMA_PREFIX}/include")
set(G4DAECHROMA_DEFINITIONS "")


