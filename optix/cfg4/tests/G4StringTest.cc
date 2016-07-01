
#include <cassert>
#include "CFG4_BODY.hh"
#include "G4String.hh"

#include "PLOG.hh"
#include "CFG4_LOG.hh"



void test_G4String(const G4String& st )
{
    LOG(info) << st ; 
}


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    CFG4_LOG_ ;


    const char* r = argv[0] ;

    std::string s = r ;

    G4String gr = r ;
    G4String gs = s ;

    LOG(info) << r ;
    LOG(info) << s ;
    LOG(info) << gr ;
    LOG(info) << gs ;


    test_G4String(r);
    test_G4String(s);
    test_G4String(gr);
    test_G4String(gs);
    





    return 0 ;
}

