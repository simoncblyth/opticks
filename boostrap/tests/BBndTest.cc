
#include <cstring>
#include "BBnd.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    const char* b0 = "Rock///Pyrex" ; 

    const char* b1 = BBnd::DuplicateOuterMaterial(b0) ;
    const char* x1 = "Rock///Rock" ; 

    assert( strcmp(b1,x1) == 0 ); 

    return 0 ; 
}
 
