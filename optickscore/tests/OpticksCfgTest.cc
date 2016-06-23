#include "Opticks.hh"
#include "OpticksCfg.hh"

#include "OKCORE_BODY.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    LOG(info) << argv[0] ;

    Opticks* opticks = new Opticks(argc, argv);

    BCfg* cfg  = new BCfg("umbrella", false) ;

    BCfg* ocfg = new OpticksCfg<Opticks>("opticks", opticks,false);

    cfg->add(ocfg);

    cfg->commandline(argc, argv);

    std::string desc = cfg->getDescString();

    LOG(info) << " desc " << desc ;


    ocfg->dump("dump");


    return 0 ; 
}
