#include <cassert>

#include "BOpticksResource.hh"
#include "BOpticksEvent.hh"
#include "OPTICKS_LOG.hh"

class BOpticksEventTest {
    public:
        void check_base_override(const char* det, const char* source, const char* tag, const char* stem, const char* ext );
        void check_layout_version(const char* det, const char* source, const char* tag, const char* stem, const char* ext );
};


void BOpticksEventTest::check_base_override(const char* det, const char* source, const char* tag, const char* stem, const char* ext )
{
    std::string p0, p1, p2  ; 

    p0 = BOpticksEvent::path(det,source,tag,stem, ext) ; 
    
    const char* gensteps_dir = BOpticksResource::GenstepsDir(); 
    BOpticksEvent::SetOverrideEventBase(gensteps_dir) ;
    p1 = BOpticksEvent::path(det,source,tag,stem, ext) ; 
    BOpticksEvent::SetOverrideEventBase(NULL) ;

    p2 = BOpticksEvent::path(det,source,tag,stem, ext) ; 

    LOG(info) << "p0 " <<  p0 ;
    LOG(info) << "p1 " <<  p1 ;
    LOG(info) << "p2 " <<  p2 ;

    assert(strcmp(p0.c_str(), p2.c_str())==0);
}

void BOpticksEventTest::check_layout_version(const char* det, const char* source, const char* tag, const char* stem, const char* ext )
{
    std::string p0, p1, p2, p3 ; 

    p0 = BOpticksEvent::path(det,source,tag,stem,ext) ;

    BOpticksEvent::SetLayoutVersion(1) ;
    p1 = BOpticksEvent::path(det,source,tag,stem,ext) ;

    BOpticksEvent::SetLayoutVersion(2) ;
    p2 = BOpticksEvent::path(det,source,tag,stem,ext) ;

    BOpticksEvent::SetLayoutVersionDefault() ;
    p3 = BOpticksEvent::path(det,source,tag,stem,ext) ;

    LOG(info) << "p0 " << p0 ; 
    LOG(info) << "p1 " << p1 ; 
    LOG(info) << "p2 " << p2 ; 
    LOG(info) << "p3 " << p3 ; 

    assert(strcmp(p0.c_str(), p3.c_str())==0);
}




void test_notag()
{
    std::string dir0 = BOpticksEvent::directory("det","source", "tag");
    std::string dir1 = BOpticksEvent::directory("det","source", NULL );
    LOG(info) 
         << " test_notag "
         << " dir0 " << dir0 
         << " dir1 " << dir1 
         ; 

}

void test_srcpath()
{
    // needs resources loaded : so cannot test at this level
 
    const char* det = "det" ; 
    const char* typ = "typ" ; 
    const char* tag = "tag" ; 
    std::string p = BOpticksEvent::srcpath(det, typ, tag ); 

    LOG(info) << p ; 

}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

/*
    BOpticksResource res ; 
    res.Summary();

    BOpticksEventTest oet ; 
    oet.check_base_override( "det",   "source","tag", "stem", "ext") ;
    oet.check_base_override( "dayabay", "cerenkov","1", "", ".npy") ;

    oet.check_layout_version("PmtInBox","torch","10", "ox", ".npy") ;
    oet.check_layout_version("det",   "source","tag", "stem", "ext") ;

    test_notag();
*/
    //test_srcpath(); 


    return 0 ; 
}
