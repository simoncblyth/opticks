// OM_TEST=

#include <cstdlib>
#include <cassert>
#include <vector>

#include "OPTICKS_LOG.hh"

#include "NBox.hpp"
#include "NSphere.hpp"
#include "NNode.hpp"
#include "NTreeAnalyse.hpp"
#include "NTreePositive.hpp"



/*
1463 def test_positivize():
1464     log.info("test_positivize")
1465 
1466     a = CSG("sphere", param=[0,0,-50,100] )
1467     b = CSG("sphere", param=[0,0, 50,100] )
1468     c = CSG("box", param=[0,0, 50,100] )
1469     d = CSG("box", param=[0,0, 0,100] )
1470     e = CSG("box", param=[0,0, 0,100] )
1471 
1472     ab = CSG("union", left=a, right=b )
1473     de = CSG("difference", left=d, right=e )
1474     cde = CSG("difference", left=c, right=de )
1475 
1476     abcde = CSG("intersection", left=ab, right=cde )
1477 
1478     abcde.analyse()
1479     print "original\n\n", abcde.txt
1480     print "operators: " + " ".join(map(CSG.desc, abcde.operators_()))
1481 
1482     abcde.positivize()
1483     print "positivize\n\n", abcde.txt
1484     print "operators: " + " ".join(map(CSG.desc, abcde.operators_()))
1485 
*/


void test_positivize()
{
    nnode* a = make_sphere(0,0,-50,100) ;  
    nnode* b = make_sphere(0,0, 50,100) ;  
    nnode* c = make_box(0,0, 50,100) ;  
    nnode* d = make_box(0,0,  0,100) ;  
    nnode* e = make_box(0,0,  0,100) ;  

    nnode* ab = nnode::make_operator( CSG_UNION, a, b );
    nnode* de = nnode::make_operator( CSG_DIFFERENCE, d, e );
    nnode* cde = nnode::make_operator( CSG_DIFFERENCE, c, de );
    nnode* abcde = nnode::make_operator( CSG_INTERSECTION, ab, cde );

    LOG(info) << abcde->desc() ; 
    LOG(info) << NTreeAnalyse<nnode>::Desc(abcde) ; 

    NTreePositive<nnode> pos(abcde) ; 
    LOG(info) << NTreeAnalyse<nnode>::Desc(abcde) ; 
}






int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    test_positivize() ; 

    return 0 ; 
}


