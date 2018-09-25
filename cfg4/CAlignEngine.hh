#pragma once

#include <string>
#include <unordered_set>
#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"
template <typename T> class NPY ; 

/**
CAlignEngine
==============

CAlignEngine isa CLHEP::HepRandomEngine, when enabled it 
arranges that G4UniformRand() will return multiple 
independant streams of pre-cooked random numbers.
Enabling and disabling are done via the static method:: 

    CAlignEngine::SetSequenceIndex(int seq_idx)   

When seq_idx is 0 or more the CAlignEngine is set as "theEngine"
and G4UniformRand() will supply the precooked sequence corresponding 
to the seq_idx.  When seq_idx is negative the engine in place previously
is restored as theEngine and ordinary non-precooked random numbers
are returned from G4UniformRand().

Notes
------

Factor out the "kernel" of CRandomEngine and  
generalize to maintain separate cursors for each stream, 
so can switch between streams, resuming as appropriate. 

**/

#include <ostream>
#include "CLHEP/Random/RandomEngine.h"

class CFG4_API CAlignEngine : public CLHEP::HepRandomEngine 
{
        friend struct CAlignEngineTest ; 
    public:
        static bool Initialize(const char* ssdir); 
        static void Finalize(); 
        static void SetSequenceIndex(int record_id); 
        double flat() ;  
    private:
        static CAlignEngine* INSTANCE ; 
        static const char* LOGNAME ; 
        static const char* InitSimLog( const char* ssdir, const char* reldir);
    private:
        CAlignEngine(const char* ssdir, const char* reldir);
        virtual ~CAlignEngine(); 

        void setSequenceIndex(int record_id); 
        std::string desc() const ; 
        bool isReady() const ; 
    private:
        const char*              m_seq_path ; 
        NPY<double>*             m_seq ; 
        double*                  m_seq_values ; 
        int                      m_seq_ni ; 
        int                      m_seq_nv ; 
        NPY<int>*                m_cur ; 
        int*                     m_cur_values ; 
        int                      m_seq_index ; 
        bool                     m_recycle ;   // temporary measure to decide on how much needs to be precooked 
        CLHEP::HepRandomEngine*  m_default ; 
        const char*              m_sslogpath ; 
        bool                     m_backtrace ; 
        std::ostream*            m_out ; 
        int                      m_count ; 
        int                      m_modulo ; 
    private:
        bool isTheEngine() const ; 
        void enable() const ; 
        void disable() const ; 
    private:
        std::string name() const ;
        void flatArray (const int size, double* vect);
        void setSeed(long , int) ; 
        void setSeeds(const long * , int) ; 
        void saveStatus( const char * ) const ; 
        void restoreStatus( const char * ); 
        void showStatus() const ; 

        std::unordered_set<unsigned> m_recycle_idx ; 
};

#include "CFG4_TAIL.hh"

 
