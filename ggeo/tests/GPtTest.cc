// TEST=GPtTest om-t

#include "OPTICKS_LOG.hh"
#include "GPt.hh"
#include "GPts.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    GPts* pts = GPts::Make() ; 

    pts->add( new GPt( 101, 10001, "red" ) );
    pts->add( new GPt( 202, 20002, "green" ) );
    pts->add( new GPt( 303, 30003, "blue" ) );

    pts->dump();  

    const char* dir = "$TMP/GGeo/GPtsTest" ; 
    pts->save(dir); 

    GPts* pts2 = GPts::Load(dir); 
    pts2->dump(); 


    return 0 ;
}

