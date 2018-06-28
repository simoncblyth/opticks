#pragma once

#include <vector>
#include <string>
#include <map>

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class Opticks ; 
class OpticksRun ; 
class OpticksEvent ; 

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





**/

#include "CLHEP/Random/RandomEngine.h"


template <typename T> class BLocSeq ; 


class CFG4_API CRandomEngine : public CLHEP::HepRandomEngine 
{
    public:
        static std::string CurrentProcessName();
        static std::string FormLocation(const char* file, int line);
    public:
        CRandomEngine(CG4* g4);
        void dumpDouble(const char* msg, double* v, unsigned width ) const  ; 
        bool hasSequence() const ; 
        const char* getPath() const ; 
    protected:
        friend class CG4 ; 
        friend struct CRandomEngineTest ; 
        void postpropagate();
        void preTrack();
        void postTrack();
        void postStep();
    private:
        void init(); 
        void initCurand(); 
        void setupCurandSequence(int record_id);

        void dump(const char* msg) const ; 
        void dumpFlat(); 

    public:
        std::string desc() const ; 
    public:
        std::string name() const ;
        double flat() ;  
        double flat_instrumented(const char* file, int line) ;  
        void flatArray (const int size, double* vect);

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
        bool                          m_dbgkludgeflatzero ; 
        OpticksRun*                   m_run ; 

        OpticksEvent*                 m_okevt ; 
        unsigned long long            m_okevt_seqhis ; 
        const char*                   m_okevt_pt ; 

        OpticksEvent*                 m_g4evt ; 

        const std::vector<unsigned>&  m_mask ;  
        bool                          m_masked ;  

        const char*              m_path ; 
        int                      m_alignlevel ; 
        long                     m_seed ; 
        bool                     m_internal ; 
        bool                     m_skipdupe ; 
        BLocSeq<unsigned long long>*  m_locseq ; 

        NPY<double>*             m_curand ; 
        int                      m_curand_index ; 
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
 
        std::vector<unsigned> m_jump_photons ; 

    private:
        void setSeed(long , int) ; 
        void setSeeds(const long * , int) ; 
        void saveStatus( const char * ) const ; 
        void restoreStatus( const char * ); 
        void showStatus() const ; 
};

#include "CFG4_TAIL.hh"

