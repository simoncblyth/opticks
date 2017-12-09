#pragma once

#include <vector>
#include <string>
#include <map>

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class Opticks ; 

class CG4 ; 
struct CG4Ctx ; 
template <typename T> class NPY ; 

/**
CRandomEngine
=====

Canonical m_engine instance is resident of CG4 and is instanciated with it, 
when the --align option is used.

**/

#include "CLHEP/Random/RandomEngine.h"

namespace CLHEP
{
    class NonRandomEngine ; 
}


template <typename T> class BLocSeq ; 


class CFG4_API CRandomEngine : public CLHEP::HepRandomEngine 
{
    public:
        static std::string FormLocation();
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
        void pretrack();
        void posttrack();
        void poststep();
    private:
        void init(); 
        void initCurand(); 
        void setupCurandSequence(int record_id);

        bool isNonRan() const ; 
        bool isDefault() const ; 
        void dump(const char* msg) const ; 
        void dumpFlat(); 

    public:
        std::string desc() const ; 
    public:
        std::string name() const ;
        double flat() ;  
        double flat_instrumented(const char* file, int line) ;  
        void flatArray (const int size, double* vect);
    private:
        CG4*                          m_g4 ; 
        CG4Ctx&                       m_ctx ; 
        Opticks*                      m_ok ; 
        const std::vector<unsigned>&  m_mask ;  
        bool                          m_masked ;  

        const char*              m_path ; 
        int                      m_alignlevel ; 
        long                     m_seed ; 
        bool                     m_internal ; 
        bool                     m_skipdupe ; 
        BLocSeq<unsigned long long>*  m_locseq ; 

        CLHEP::HepJamesRandom*   m_james ; 
        CLHEP::NonRandomEngine*  m_nonran ; 
        CLHEP::HepRandomEngine*  m_engine ; 
        NPY<double>*             m_curand ; 
        int                      m_curand_ni ; 
        int                      m_curand_nv ; 
        int                      m_current_record_flat_count ; 
        double                   m_flat ; 

        std::string              m_location ; 

    private:
        void setSeed(long , int) ; 
        void setSeeds(const long * , int) ; 
        void saveStatus( const char * ) const ; 
        void restoreStatus( const char * ); 
        void showStatus() const ; 
};

#include "CFG4_TAIL.hh"

