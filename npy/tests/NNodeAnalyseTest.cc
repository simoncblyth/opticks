#include "OPTICKS_LOG.hh"

#include "No.hpp"
#include "NNodeCollector.hpp"
#include "NNodeAnalyse.hpp"

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

    NNodeAnalyse<no> ana(a); 
    ana.nodes->dump() ; 

    LOG(info) << ana.desc() ; 


    return 0 ; 
}


