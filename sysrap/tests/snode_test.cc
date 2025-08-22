// ~/o/sysrap/tests/snode_test.sh

#include "snode.h"
#include <cassert>

int main()
{
    snode n = {} ;
    int lvid_idx = snode_field::Idx("LVID") ;
    int copyno_idx = snode_field::Idx("COPYNO") ;
    int depth_idx = snode_field::Idx("DEPTH") ;

    n.lvid = 101 ;
    n.copyno = 50000 ;

    int lvid = n.get_attrib(lvid_idx);
    int copyno = n.get_attrib(copyno_idx);
    int depth = n.get_attrib(depth_idx);

    assert( lvid == n.lvid );
    assert( copyno == n.copyno );
    assert( depth == 0 );

    return 0;
}
