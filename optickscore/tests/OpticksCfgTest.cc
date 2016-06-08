#include "Opticks.hh"
#include "OpticksCfg.hh"

#include "BLog.hh"

int main(int argc, char** argv)
{
    Opticks* opticks = new Opticks(argc, argv);

    BCfg* cfg  = new BCfg("umbrella", false) ;

    BCfg* ocfg = new OpticksCfg<Opticks>("opticks", opticks,false);

    cfg->add(ocfg);

    cfg->commandline(argc, argv);

    LOG(info) << cfg->getDesc() ;


    ocfg->dump("dump");


    return 0 ; 
}
