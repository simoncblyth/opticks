#include "OPTICKS_LOG.hh"
#include "no.hpp"
#include "NGrid.hpp"


int main(int argc, char** argv)
{
    OPTICKS_LOG_COLOR__(argc, argv);

    NGrid<no> g(10,10)     ;

    no* a = new no {"a", NULL, NULL }; 
    no* b = new no {"b", NULL, NULL };
    no* c = new no {"c", NULL, NULL };

    g.set(0,0,a) ;
    g.set(4,4,b) ;
    g.set(9,9,c) ;

    LOG(info) << "\n" << g.desc() ; 

    return 0 ; 
}
