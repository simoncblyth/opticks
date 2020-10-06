/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#pragma once

class OpticksHub ; 
class OpticksGun ; 
class Opticks ; 
template <typename T> class NPY  ; 
template <typename T> class OpticksCfg ; 
class NLookup ; 
class NCSG ; 

class GGeo ; 
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

m_gen pointers are available in the principal users

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
        NPY<float>*          makeLegacyGensteps(unsigned gencode);
        NPY<float>*          loadLegacyGenstepFile(const char* label);
        TorchStepNPY*        makeTorchstep(unsigned gencode);
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
        GGeo*                 m_ggeo ; 
        GGeoBase*             m_ggb ; 
        GBndLib*              m_blib ; 

        NLookup*              m_lookup ; 
        TorchStepNPY*         m_torchstep ;
        FabStepNPY*           m_fabstep ;  
        NCSG*                 m_csg_emit ; 
        bool                  m_dbgemit ;   // --dbgemit
        NEmitPhotonsNPY*      m_emitter ; 
        NPY<float>*           m_input_photons ; 
        unsigned              m_tagoffset ;  
        NPY<float>*           m_direct_gensteps ; 
        NPY<float>*           m_legacy_gensteps ; 
        unsigned              m_source_code ; 
  
};


