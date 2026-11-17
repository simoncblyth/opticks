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
    static s_RPN* populate();

    static int init();
    static int oper();
    static int save();
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



int srpn_test::oper()
{
    srpn nd = {} ;
    for(unsigned i=0 ; i <= 0x7ffu ; i++ )
    {
        unsigned lhs = i ;
        unsigned rhs = 0x7ffu - i ;
        nd.set_opernode(CSG_UNION, lhs, rhs);
        assert( nd.leftspan() == lhs );
        assert( nd.rightspan() == rhs );
    }
    std::cout << nd.desc() << "\n" ;
    return 0;
}


s_RPN* srpn_test::populate()
{
    s_RPN* rpn = new s_RPN ;
    {
        sparm pa( 100.f, 0.f, 0.f, 0.f, 0.f, 0.f ) ;
        qat4  it( 0.f, 0.f, 100.f );
        rpn->add_primnode( CSG_SPHERE, false, it.is_identity(), pa, it );
    }
    {
        sparm pa( 100.f, 0.f, 0.f, 0.f, 0.f, 0.f ) ;
        qat4  it( 0.f, 0.f, 50.f );
        rpn->add_primnode( CSG_SPHERE, false, it.is_identity(), pa, it );
    }
    rpn->add_opernode( CSG_UNION, 1, 1 );
    return rpn ;
}



int srpn_test::save()
{
    const char* rel = "rpn";

    s_RPN* rpn0 = populate();
    rpn0->save("$FOLD", rel);
    std::cout << "rpn0\n" << rpn0->desc() << "\n\n" ;

    s_RPN* rpn1 = new s_RPN ;
    rpn1->load("$FOLD", rel);
    std::cout << "rpn1\n" << rpn1->desc() << "\n\n" ;

    return 0 ;
}



int srpn_test::main()
{
    const char* TEST = ssys::getenvvar("TEST", "ALL");
    bool ALL = strcmp(TEST, "ALL") == 0 ;

    int rc = 0 ;
    if(ALL||0==strcmp(TEST, "init"))    rc += init();
    if(ALL||0==strcmp(TEST, "save"))    rc += save();
    if(ALL||0==strcmp(TEST, "oper"))    rc += oper();

    return rc ;
}


int main()
{
    return srpn_test::main();
}


