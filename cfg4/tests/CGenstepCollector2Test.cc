/**

::

    CGenstepCollector2Test 
    EVTSKIP=1 CGenstepCollector2Test 


**/

#include <iostream>
#include "SSys.hh"
#include "OPTICKS_LOG.hh"
#include "Opticks.hh"
#include "CGenstepCollector.hh"
#include "CManager.hh"

struct CGenstepCollector2Test 
{
    const NLookup*     lookup ; 
    CManager*          manager ;  
    CGenstepCollector* collector ; 
    bool               evtskip ; 

    CGenstepCollector2Test(Opticks* ok, bool evtskip_)
        :
        lookup(nullptr), 
        manager(new CManager(ok)), 
        collector(new CGenstepCollector(lookup)),
        evtskip(evtskip_)
    {
    }

    void add()
    {
        const G4Event* event = nullptr ; 

        if(!evtskip)
        manager->BeginOfEventAction(event); 

        CGenstep gs ;  
        for(unsigned i=0 ; i < 10 ; i++)
        { 
            gs = collector->addGenstep(100*(i+1), 'C'); 
            std::cout << gs.desc() << std::endl ; 
        }

        if(!evtskip)
        manager->EndOfEventAction(event); 
    }


};

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    Opticks ok(argc, argv); 
    ok.configure(); 
    LOG(info) << " ok.isSave " << ok.isSave() ; 
    ok.setSave(true); 
    LOG(info) << " ok.isSave " << ok.isSave() ; 

    ok.setSpaceDomain(0.f, 0.f, 0.f, 1000.f ); 

    bool evtskip = SSys::getenvbool("EVTSKIP"); 
    CGenstepCollector2Test t(&ok, evtskip) ; 
    t.add(); 

    return 0 ; 
}
