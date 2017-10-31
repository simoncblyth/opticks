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
class NEmitPhotonsNPY ; 

#include "OKGEO_API_EXPORT.hh"

/*
OpticksGen
============

High level genstep control.
Canonical m_gen instance is member of ok/OKMgr OR okg4/OKG4Mgr
which is instanciated by OpticksHub::init after the geometry
has been loaded.

*/


class OKGEO_API OpticksGen 
{ 
        friend class OpMgr ;  // for setInputGensteps
    public:
        OpticksGen(OpticksHub* hub);
    public:
        NPY<float>*          getInputPhotons() const ;    // currently only used for NCSG emitter testing 
        NPY<float>*          getInputGensteps() const ;
        TorchStepNPY*        getTorchstep();
    public:
        FabStepNPY*          makeFabstep();  
    private:
        void                 init();
        void                 initFromGensteps();
        void                 initFromEmitter();
    private:
        NPY<float>*          makeInputGensteps(unsigned code);
        NPY<float>*          loadGenstepFile(const char* label);
        TorchStepNPY*        makeTorchstep();
    private:
       //  FabStepNPY and TorchStepNPY are specializations of GenstepNPY
        void                 targetGenstep( GenstepNPY* gs );
        void                 setMaterialLine( GenstepNPY* gs );
    private:
        void                 setInputGensteps(NPY<float>* igs);
        void                 setInputPhotons(NPY<float>* iox);
    private:
        OpticksHub*           m_hub ; 
        Opticks*              m_ok ; 
        OpticksCfg<Opticks>*  m_cfg ; 
        NLookup*              m_lookup ; 
        GGeo*                 m_ggeo ;  
        TorchStepNPY*         m_torchstep ;
        FabStepNPY*           m_fabstep ;  
        NPY<float>*           m_input_gensteps ; 
        NCSG*                 m_csg_emit ; 
        NEmitPhotonsNPY*      m_emitter ; 
        NPY<float>*           m_input_photons ; 
};


