#include <iostream>
#include "OPTICKS_LOG.hh"
#include "Opticks.hh"
#include "CGenstepCollector.hh"
#include "CManager.hh"

struct CGenstepCollector2Test 
{
    CGenstepCollector2Test()
    {
        NLookup* lookup = nullptr ; 
        CGenstepCollector gsc(lookup) ; 

        CGenstep gs ;  
        for(unsigned i=0 ; i < 10 ; i++)
        { 
            gs = gsc.addGenstep(100*(i+1), 'C'); 
            std::cout << gs.desc() << std::endl ; 
        }
    }
};

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    Opticks ok(argc, argv); 
    ok.configure(); 
    LOG(info) << " ok.isSave " << ok.isSave() ; 
    CManager mgr(&ok); 
    CGenstepCollector2Test gsct ; 
    return 0 ; 
}
