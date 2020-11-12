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

#include <vector>
#include <string>
#include <map>

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"
#include "plog/Severity.h"

class Opticks ; 
class OpticksRun ; 
class OpticksEvent ; 

class BLog ; 
class CG4 ; 
struct CG4Ctx ; 
template <typename T> class NPY ; 

/**
CRandomEngine
===============

CRandomEngine isa CLHEP::HepRandomEngine which gets annointed
as the Geant4 engine in CRandomEngine::init with 
CLHEP::HepRandom::setTheEngine.

Canonical m_engine instance is resident of CG4 and is instanciated with it, 
when the --align option is used.

With the engine instanciated standard G4UniformRand calls to get random numbers
are routed via the engine which provides values from precooked sequences generated
by curand for each photon record_id on GPU. 

To provide the appropriate sequence of random numbers for the current photon
it is necessary to call CRandomEngine::preTrack  and the m_ctx referenced needs 
to have the photon record id.

Note that nothing special is needed on the GPU side for alignment, which just uses
standard curand to get its randoms. 
During development the below optional macros were used for dumping the random consumption.
GPU dumping is only comprehensible when restricting to single photons.

ALIGN_DEBUG 
WITH_ALIGN_DEV_DEBUG 

**/

#include "CLHEP/Random/RandomEngine.h"
#include "CRandomListener.hh"

template <typename T> class BLocSeq ; 


#define DYNAMIC_CURAND 1


#ifdef DYNAMIC_CURAND
template <typename T> class TCURAND ; 
#endif


class CFG4_API CRandomEngine : public CRandomListener, public CLHEP::HepRandomEngine 
{
        static const char* TMPDIR ; 
        static const plog::Severity LEVEL ; 
    public:
        static std::string CurrentGeant4ProcessName();
        static std::string FormLocation(const char* file, int line);
        static const char* PindexLogPath(unsigned mask_index);
    public:
        CRandomEngine(CG4* g4);
        void dumpDouble(const char* msg, double* v, unsigned width ) const  ; 
        bool hasSequence() const ; 

#ifdef DYNAMIC_CURAND
#else
        const char* getPath() const ; 
#endif
    protected:
        friend class CG4 ; 
        friend struct CRandomEngineTest ; 

        // CRandomListener
        void postpropagate();
        void preTrack();
        void postTrack();
        void postStep();
    private:
        int  precurand(); 
        int  postcurand(); 
        int  preinit(); 
        void init(); 
        void initCurand(); 
        void run_ucf_script(unsigned mask_index) ; 
        void dumpPindexLog(const char* msg);

        void checkTranche(); 
        void dumpTranche(); 
#ifdef DYNAMIC_CURAND
        void setupTranche(int tranche_id); 
#endif
        void setupCurandSequence(int record_id);

        void dump(const char* msg) const ; 
        void dbgFlat(); 
        void compareLogs(const char* msg);

    public:
        void addNote(const char* note, int value); 
        void addCut( const char* ckey, double cvalue); 
        std::string desc() const ; 
    public:
        std::string name() const ;
        double flat() ;  
        double flat_instrumented(const char* file, int line) ;  
        void flatArray (const int size, double* vect);
        int    getCursor() const ; 
        int    getCurrentStepFlatCount() const ; 
        int    getCurrentRecordFlatCount() const ; 
    public:
        void setRandomSequence(double* s, int n);
        int  findIndexOfValue(double s, double tolerance=1e-6) ; 
        void jump(int offset); 
        double _flat(); 
        double _peek(int offset) const  ; // does not increment anything, just looks around
    private:
        CG4*                          m_g4 ; 
        CG4Ctx&                       m_ctx ; 
        Opticks*                      m_ok ; 
        bool                          m_dbgflat ;   
        int                           m_curflatsigint ; 
        int                           m_preinit ;  
        bool                          m_dbgkludgeflatzero ; 
        OpticksRun*                   m_run ; 

        OpticksEvent*                 m_okevt ; 
        unsigned long long            m_okevt_seqhis ; 
        const char*                   m_okevt_pt ; 
        const char*                   m_pindexlogpath ; 
        BLog*                         m_pindexlog ;  
        BLog*                         m_dbgflatlog ;  

        OpticksEvent*                 m_g4evt ; 

        const std::vector<unsigned>&  m_mask ;  
        bool                          m_masked ;  

        int                      m_alignlevel ; 
        long                     m_seed ; 
        bool                     m_internal ; 
        bool                     m_skipdupe ; 
        BLocSeq<unsigned long long>*  m_locseq ; 
        int                      m_tranche_size ; 
        int                      m_tranche_id ; 
        int                      m_tranche_ibase ; 
        int                      m_tranche_index ; 
#ifdef DYNAMIC_CURAND
        int                      m_precurand ; 
        TCURAND<double>*         m_tcurand ; 
        int                      m_postcurand ; 
#else
        const char*              m_path ; 
#endif
        NPY<double>*             m_curand ; 
        int                      m_curand_ni ; 
        int                      m_curand_nv ; 
        int                      m_current_record_flat_count ; 
        int                      m_current_step_flat_count ; 
        int                      m_jump ;
        int                      m_jump_count ;  
        double                   m_flat ; 

        std::string              m_location ; 

    private:
       
        std::vector<double> m_sequence ; 
        int                 m_cursor; 
        int                 m_cursor_old ;
        unsigned            m_setupTranche_acc ;  
 
        std::vector<unsigned> m_jump_photons ; 
        std::vector<int>      m_step_cursors ; 
      

    private:
        void setSeed(long , int) ; 
        void setSeeds(const long * , int) ; 
        void saveStatus( const char * ) const ; 
        void restoreStatus( const char * ); 
        void showStatus() const ; 
};

#include "CFG4_TAIL.hh"

