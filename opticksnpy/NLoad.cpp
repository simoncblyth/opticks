#include "BOpticksEvent.hh"
#include "BOpticksResource.hh"

#include "NLoad.hpp"
#include "NPY.hpp"

NPY<float>* NLoad::Gensteps(const char* det, const char* typ, const char* tag)
{
    const char* gensteps_dir = BOpticksResource::GenstepsDir();  // eg /usr/local/opticks/opticksdata/gensteps
    NPY<float>* npy = NULL ; 

    BOpticksEvent::SetOverrideEventBase(gensteps_dir) ;
    BOpticksEvent::SetLayoutVersion(1) ;     

    const char* stem = "" ; // backward compat stem of gensteps
    std::string path = BOpticksEvent::path(det, typ, tag, stem, ".npy");
    npy = NPY<float>::load(path.c_str()) ;

    BOpticksEvent::SetOverrideEventBase(NULL) ;
    BOpticksEvent::SetLayoutVersionDefault() ;

    return npy ; 
}


