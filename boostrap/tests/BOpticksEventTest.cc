#include <cassert>

#include "BOpticksResource.hh"
#include "BOpticksEvent.hh"
#include "BRAP_LOG.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    BRAP_LOG__ ; 

    BOpticksResource res ; 

    res.Summary();

    BOpticksEvent evt ; 


    std::string path_1 = BOpticksEvent::path("typ","tag", "det") ; 

    const char* gensteps_dir = BOpticksResource::GenstepsDir(); 
    BOpticksEvent::SetOverrideEventBase(gensteps_dir) ;
    std::string path_2 = BOpticksEvent::path("typ","tag", "det") ; 
    BOpticksEvent::SetOverrideEventBase(NULL) ;


    std::string path_3 = BOpticksEvent::path("typ","tag", "det") ; 


    LOG(info) << "(1) BOpticksEvent::path(\"typ\",\"tag\",\"det\") = " <<  path_1 ;
    LOG(info) << "(2) BOpticksEvent::path(\"typ\",\"tag\",\"det\") = " <<  path_2 ;
    LOG(info) << "(3) BOpticksEvent::path(\"typ\",\"tag\",\"det\") = " <<  path_3 ;

    assert(strcmp(path_1.c_str(), path_3.c_str())==0);

    return 0 ; 
}
