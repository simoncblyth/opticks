#include "BOpticksEvent.hh"
#include "BOpticksResource.hh"

#include "NLoad.hpp"
#include "NPY.hpp"


std::string NLoad::GenstepsPath(const char* det, const char* typ, const char* tag)
{
    const char* gensteps_dir = BOpticksResource::GenstepsDir();  // eg /usr/local/opticks/opticksdata/gensteps
    BOpticksEvent::SetOverrideEventBase(gensteps_dir) ;
    BOpticksEvent::SetLayoutVersion(1) ;     

    const char* stem = "" ; // backward compat stem of gensteps
    std::string path = BOpticksEvent::path(det, typ, tag, stem, ".npy");

    BOpticksEvent::SetOverrideEventBase(NULL) ;
    BOpticksEvent::SetLayoutVersionDefault() ;

    return path ; 
}

NPY<float>* NLoad::Gensteps(const char* det, const char* typ, const char* tag)
{
    std::string path = GenstepsPath(det, typ, tag);
    NPY<float>* gs = NPY<float>::load(path.c_str()) ;
    return gs ; 
}


std::string NLoad::directory(const char* det, const char* typ, const char* tag, const char* anno)
{
   std::string tagdir = BOpticksEvent::directory(det, typ, tag, anno ? anno : NULL );  
   return tagdir ; 
}

std::string NLoad::reldir(const char* det, const char* typ, const char* tag )
{
   std::string rdir = BOpticksEvent::reldir(det, typ, tag );  
   return rdir ; 
}


