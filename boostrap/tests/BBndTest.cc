
#include <cstring>
#include "BBnd.hh"
#include "PLOG.hh"



void test_DuplicateOuterMaterial()
{
    const char* b0 = "Rock///Pyrex" ; 

    const char* b1 = BBnd::DuplicateOuterMaterial(b0) ;
    const char* x1 = "Rock///Rock" ; 

    if( strcmp(b1,x1) != 0 )
    {
        LOG(error) << " b1 [" << b1 << "]"
                   << " x1 [" << x1 << "]"
                   ;
    }
    assert( strcmp(b1,x1) == 0 ); 
}


void test_BBnd()
{
    {
        BBnd b("omat/osur/isur/imat");
        LOG(info) << b.desc() ; 
        assert( b.omat && b.osur && b.isur && b.imat );  
    }

    {
        BBnd b("omat///imat");
        LOG(info) << b.desc() ; 
        assert( b.omat && !b.osur && !b.isur && b.imat );  
    }
   
}


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    test_DuplicateOuterMaterial();
    test_BBnd();


    return 0 ; 
}
 
