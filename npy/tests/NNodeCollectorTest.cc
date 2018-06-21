#include "OPTICKS_LOG.hh"

#include "No.hpp"
#include "NNodeCollector.hpp"

int main(int argc, char** argv)
{
    OPTICKS_LOG_COLOR__(argc, argv);

    no* g = new no {"g", NULL, NULL } ;
    no* f = new no {"f", NULL, NULL } ;
    no* e = new no {"e", NULL, NULL } ;
    no* d = new no {"d", NULL, NULL } ;
    no* c = new no {"c",  f,  g } ;
    no* b = new no {"b",  d,  e } ;
    no* a = new no {"a",  b,  c } ; 
   
    LOG(info) << a->desc() ; 

    NNodeCollector<no> nc(a); 

    assert( nc.inorder.size() == 7 ); 
    assert( nc.postorder.size() == 7 ); 
    assert( nc.preorder.size() == 7 ); 

    nc.dump();


    return 0 ; 
}



