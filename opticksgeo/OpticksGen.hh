#pragma once

class OpticksHub ; 
class Opticks ; 
template <typename T> class NPY  ; 
template <typename T> class OpticksCfg ; 
class NLookup ; 
class GGeo ; 

class GenstepNPY ; 
class TorchStepNPY ; 
class FabStepNPY ; 

#include "OKGEO_API_EXPORT.hh"

/*
OpticksGen
============

High level genstep control.
Canonical m_gen instance is member of ok/OKMgr OR okg4/OKG4Mgr


*/


class OKGEO_API OpticksGen 
{ 
    public:
        OpticksGen(OpticksHub* hub);
    public:
        NPY<float>*          getInputGensteps();
        TorchStepNPY*        getTorchstep();
    public:
        FabStepNPY*          makeFabstep();  
    private:
        void                 init();
        void                 initInputGensteps();
    private:
        NPY<float>*          loadGenstepFile(const char* label);
        TorchStepNPY*        makeTorchstep();
    private:
        void                 targetGenstep( GenstepNPY* gs );
        void                 setMaterialLine( GenstepNPY* gs );
    private:
        void                 setInputGensteps(NPY<float>* igs);
    private:
        OpticksHub*           m_hub ; 
        Opticks*              m_ok ; 
        OpticksCfg<Opticks>*  m_cfg ; 
        NLookup*              m_lookup ; 
        GGeo*                 m_ggeo ;  
        TorchStepNPY*         m_torchstep ;
        FabStepNPY*           m_fabstep ;  
        NPY<float>*           m_input_gensteps ; 
};


