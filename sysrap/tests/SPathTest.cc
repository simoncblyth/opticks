// om-;TEST=SPathTest om-t 

#include <cassert>
#include <string>
#include "SPath.hh"

#include "OPTICKS_LOG.hh"


void test_Stem()
{
    const char* name = "hello.cu" ; 
    const char* stem = SPath::Stem(name); 
    const char* x_stem = "hello" ; 
    assert( strcmp( stem, x_stem ) == 0 ); 
}


int main(int argc , char** argv )
{
    OPTICKS_LOG(argc, argv);

    test_Stem();  

    return 0  ; 
}

