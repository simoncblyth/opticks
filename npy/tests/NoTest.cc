#include "OPTICKS_LOG.hh"

#include "No.hpp"
#include "NNodeCollector.hpp"
#include "NTreeAnalyse.hpp"

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
   
    LOG(info) << "a.desc" << std::endl << a->desc() ; 
    NTreeAnalyse<no> ana(a); 

    LOG(info) << "ana.nodes.dump" ; 
    ana.nodes->dump() ; 

    LOG(info) << "ana.desc" ; 
    LOG(info) << ana.desc() ; 

    LOG(info) << "make_deepcopy" ; 
    no* a_copy = a->make_deepcopy(); 
    assert( a_copy );  

 
    return 0 ; 
}


