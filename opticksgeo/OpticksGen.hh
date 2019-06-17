#pragma once

class OpticksHub ; 
class OpticksGun ; 
class Opticks ; 
template <typename T> class NPY  ; 
template <typename T> class OpticksCfg ; 
class NLookup ; 
class NCSG ; 
class GGeoBase ; 
class GBndLib ; 

class GenstepNPY ; 
class TorchStepNPY ; 
class FabStepNPY ; 
class NEmitPhotonsNPY ; 

#include "OKGEO_API_EXPORT.hh"
#include "plog/Severity.h"

/**
OpticksGen : High level genstep control
==========================================

Canonical instance m_gen is member of OpticksHub
and is instanciated by OpticksHub::init after 
geometry has been loaded or adopted.

m_gen copies are available in the principal users

* okop/OpMgr
* ok/OKMgr 
* okg4/OKG4Mgr

**/


class OKGEO_API OpticksGen 
{ 
        static const plog::Severity LEVEL ; 
        friend class OpMgr ;  // for setInputGensteps
        friend class OpticksHub ; // for getTorchstep getGenstepNPY getG4GunConfig
        friend class CGenerator ; // for getTorchstep getGenstepNPY getG4GunConfig
        friend struct  OpSeederTest ; // for makeFabstep 
    public:
        OpticksGen(OpticksHub* hub);
    public:
        unsigned             getSourceCode() const ;
    public:
        Opticks*             getOpticks() const ; 
        NPY<float>*          getInputPhotons() const ;    // currently only used for NCSG emitter testing 
        NPY<float>*          getInputGensteps() const ;

    private:
        FabStepNPY*          getFabStep() const  ;
        TorchStepNPY*        getTorchstep() const ;
        GenstepNPY*          getGenstepNPY() const ;
        std::string          getG4GunConfig() const ;
    private:
        NEmitPhotonsNPY*     getEmitter() const ;
    private:
        FabStepNPY*          makeFabstep();  
    private:
        void                 init();
        unsigned             initSourceCode() const ;
        void                 initFromLegacyGensteps();
        void                 initFromDirectGensteps();
        void                 initFromEmitterGensteps();
    private:
        NPY<float>*          makeLegacyGensteps(unsigned code);
        NPY<float>*          loadLegacyGenstepFile(const char* label);
        TorchStepNPY*        makeTorchstep();
    private:
        //  FabStepNPY and TorchStepNPY are specializations of GenstepNPY
        void                 targetGenstep( GenstepNPY* gs );
        void                 setMaterialLine( GenstepNPY* gs );
    private:
        void                 setLegacyGensteps(NPY<float>* igs);
        void                 setInputPhotons(NPY<float>* iox);
    private:
        OpticksHub*           m_hub ; 
        OpticksGun*           m_gun ; 
        Opticks*              m_ok ; 
        OpticksCfg<Opticks>*  m_cfg ; 
        GGeoBase*             m_ggb ; 
        GBndLib*              m_blib ; 

        NLookup*              m_lookup ; 
        TorchStepNPY*         m_torchstep ;
        FabStepNPY*           m_fabstep ;  
        NCSG*                 m_csg_emit ; 
        bool                  m_dbgemit ;   // --dbgemit
        NEmitPhotonsNPY*      m_emitter ; 
        NPY<float>*           m_input_photons ; 
        NPY<float>*           m_direct_gensteps ; 
        NPY<float>*           m_legacy_gensteps ; 
        unsigned              m_source_code ; 
  
};


