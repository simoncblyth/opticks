#include "BOpticksEvent.hh"
#include "BOpticksResource.hh"

#include "NLoad.hpp"
#include "NPY.hpp"
#include "PLOG.hh"

const plog::Severity NLoad::LEVEL = info ; 

std::string NLoad::GenstepsPath(const char* det, const char* typ, const char* tag)
{
    const char* gensteps_dir = BOpticksResource::GenstepsDir();  // eg /usr/local/opticks/opticksdata/gensteps
    BOpticksEvent::SetOverrideEventBase(gensteps_dir) ;
    BOpticksEvent::SetLayoutVersion(1) ;     

    LOG(LEVEL) 
         << " gensteps_dir " << gensteps_dir ; 
         ; 

    const char* pfx = NULL ; 
    const char* stem = NULL ; 
    std::string path = BOpticksEvent::path(pfx, det, typ, tag, stem, ".npy");

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


std::string NLoad::directory(const char* pfx, const char* det, const char* typ, const char* tag, const char* anno)
{
   std::string tagdir = BOpticksEvent::directory(pfx, det, typ, tag, anno ? anno : NULL );  
   return tagdir ; 
}

std::string NLoad::reldir(const char* pfx, const char* det, const char* typ, const char* tag )
{
   std::string rdir = BOpticksEvent::reldir(pfx, det, typ, tag );  
   return rdir ; 
}


