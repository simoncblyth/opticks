#include <cassert>

#include "GBuffer.hh"
#include "GArray.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ;


    LOG(info) << argv[0] ;

    float v[3] ;
    v[0] = 1.f ; 
    v[1] = 1.f ; 
    v[2] = 1.f ; 

    GArray<float>* a = new GArray<float>(3, v );
    assert( a->getLength() == 3 );


    const char* path = "$TMP/GArrayTest.npy" ;
    LOG(info) << "saving to " << path ; 

    a->save<float>(path);




    return 0 ;
}

