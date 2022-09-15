// name=axes_test ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include <cstring>
#include <cassert>

int main()
{
    const char* AXES = "XYZ" ; 
    unsigned x = strlen(AXES) > 0 ? AXES[0] - 'X' : ~0u ; 
    unsigned y = strlen(AXES) > 1 ? AXES[1] - 'X' : ~0u ; 
    unsigned z = strlen(AXES) > 2 ? AXES[2] - 'X' : ~0u ; 
    unsigned w = strlen(AXES) > 3 ? AXES[3] - 'X' : ~0u ; 
 
    assert( x == 0u ); 
    assert( y == 1u ); 
    assert( z == 2u ); 
    assert( w == ~0u ); 

    return 0 ; 
}
