// TEST=OpticksCfgTest om-t

#include "NSnapConfig.hpp"

#include "Opticks.hh"
#include "OpticksCfg.hh"

#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    LOG(info) << argv[0] ;

    Opticks* ok = new Opticks(argc, argv);

    BCfg* cfg  = new BCfg("umbrella", false) ;

    BCfg* ocfg = new OpticksCfg<Opticks>("opticks", ok,false);

    cfg->add(ocfg);

    cfg->commandline(argc, argv);

    std::string desc = cfg->getDescString();

    LOG(info) << "desc... " << desc ;

    LOG(info) << "ocfg... "  ;

    ocfg->dump("dump");

    LOG(info) << "sc... "  ;

    NSnapConfig* sc = ok->getSnapConfig();
    sc->dump("SnapConfig");






    //const std::string& csgskiplv = ocfg->getCSGSkipLV(); 
    //LOG(info) << " csgskiplv " << csgskiplv ; 



    LOG(info) << "DONE "  ;



    return 0 ; 
}
