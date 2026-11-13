/**
~/o/sysrap/tests/srpn_test.sh
**/

#include <vector>
#include "ssys.h"

#include "scuda.h"
#include "squad.h"
#include "srpn.h"

#include "s_RPN.h"


struct srpn_test
{
    static int init();
    static int main();
};


int srpn_test::init()
{
    std::vector<srpn> tape(4) ;

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

    tape[3].set_primnode(0x7fu, true, true, 0xffffu );
    assert( tape[3].typecode() == 0x7fu );


    for(size_t i=0 ; i < tape.size() ; i++ ) std::cout << tape[i].desc() << "\n" ;
    return 0 ;
}



int srpn_test::main()
{
    const char* TEST = ssys::getenvvar("TEST", "ALL");
    bool ALL = strcmp(TEST, "ALL") == 0 ;

    int rc = 0 ;
    if(ALL||0==strcmp(TEST, "init"))    rc += init();

    return rc ;
}


int main()
{
    return srpn_test::main();
}


