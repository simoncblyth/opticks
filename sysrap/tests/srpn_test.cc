#include <vector>
#include "srpn.h"

int main()
{
    std::vector<srpn> tape(3) ;

    tape[0].set_opernode(CSG_UNION, 0x7ffu, 0x7ffu );
    assert( tape[0].typecode() == CSG_UNION );
    assert( tape[0].leftspan() == 0x7ffu );
    assert( tape[0].rightspan() == 0x7ffu );

    tape[1].set_listnode(CSG_DISCONTIGUOUS, 0x3ffu );
    assert( tape[1].typecode() == CSG_DISCONTIGUOUS );
    assert( tape[1].listprim() == 0x3ffu );

    tape[2].set_primnode(CSG_SPHERE, true, true, 0xffffu );
    assert( tape[2].typecode() == CSG_SPHERE );
    assert( tape[2].complement() == true );
    assert( tape[2].identity() == true );
    assert( tape[2].index() == 0xffffu );

    for(size_t i=0 ; i < tape.size() ; i++ ) std::cout << tape[i].desc() << "\n" ;
    return 0 ;
}


