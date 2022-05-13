#include "GGeo.hh"
#include "GGeoGen.hh"
#include "OpticksGenstep.h"
#include "OpticksGenstep.hh"
#include "TorchStepNPY.hpp"
#include "NPY.hpp"
#include "NGLM.hpp"
#include "PLOG.hh"

const plog::Severity GGeoGen::LEVEL = PLOG::EnvLevel("GGeoGen", "DEBUG"); 

GGeoGen::GGeoGen(const GGeo* ggeo)
    :
    m_ggeo(ggeo)
{
}

const OpticksGenstep* GGeoGen::createDefaultTorchStep(unsigned num_photons, int node_index, unsigned originTrackID) const 
{
    unsigned gentype = OpticksGenstep_TORCH  ; 
    unsigned num_step = 1 ;  

    const char* config = NULL ;    
    // encompasses a default number of photons, distribution, polarization

    assert( OpticksGenstep_::IsTorchLike(gentype) ); 

    LOG(LEVEL) << " gentype " << gentype ; 

    TorchStepNPY* ts = new TorchStepNPY(gentype, config);
    ts->setOriginTrackID(originTrackID); 

    if(node_index == -1)
    {    
        node_index = m_ggeo->getFirstNodeIndexForGDMLAuxTargetLVName()  ;
    }    

    if(node_index == -1)
    {    
        LOG(error) << " failed to find target node_index " << node_index << " (reset to zero) " ;  
        node_index = 0 ;  
    }    

    glm::mat4 frame_transform = m_ggeo->getTransform( node_index ); 
    ts->setFrameTransform(frame_transform);

    for(unsigned i=0 ; i < num_step ; i++) 
    {    
        if(num_photons > 0) ts->setNumPhotons(num_photons);  // otherwise use default
        ts->addStep(); 
    }    

    NPY<float>* arr = ts->getNPY(); 

    //arr->save("$TMP/debugging/collectDefaultTorchStep/gs.npy");  

    const OpticksGenstep* ogs = new OpticksGenstep(arr); 

    return ogs ; 
}


