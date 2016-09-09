#pragma once

class OpticksHub ; 
class Opticks ; 
template <typename T> class NPY  ; 
template <typename T> class OpticksCfg ; 
class NLookup ; 
class GGeo ; 

class TorchStepNPY ; 


#include "OKGEO_API_EXPORT.hh"

class OKGEO_API OpticksGen 
{ 
    public:
        OpticksGen(OpticksHub* hub);
    public:
        NPY<float>*          getInputGensteps();
        TorchStepNPY*        getTorchstep();
    private:
        void                 init();
        void                 initInputGensteps();
        NPY<float>*          loadGenstepFile();
        TorchStepNPY*        makeTorchstep();
        void                 setInputGensteps(NPY<float>* igs);
    private:
        OpticksHub*           m_hub ; 
        Opticks*              m_ok ; 
        OpticksCfg<Opticks>*  m_cfg ; 
        NLookup*              m_lookup ; 
        GGeo*                 m_ggeo ;  
        TorchStepNPY*         m_torchstep ; 
        NPY<float>*           m_input_gensteps ; 
};


