#include "BConfig.hh"
#include "PLOG.hh"
#include "NLODConfig.hpp"


const char* NLODConfig::instanced_lodify_onload_ = ">0 : Apply LODification (GMergedMesh::MakeLODComposite) on loading non-global GMergedMesh in GGeoLib " ;  


NLODConfig::NLODConfig(const char* cfg)  
    :
    bconfig(new BConfig(cfg)),
    verbosity(0),
    levels(3),
    instanced_lodify_onload(0)
{
    LOG(verbose) << "NLODConfig::NLODConfig"
              << " cfg [" << ( cfg ? cfg : "NULL" ) << "]"
              ;

    // TODO: incorp the help strings into the machinery and include in dumping 

    bconfig->addInt("verbosity", &verbosity );
    bconfig->addInt("levels", &levels );
    bconfig->addInt("levels", &levels );
    bconfig->addInt("instanced_lodify_onload", &instanced_lodify_onload );

    bconfig->parse();
}

void NLODConfig::dump(const char* msg) const
{
    bconfig->dump(msg);
}

