#pragma once

#include <vector>
#include <string>
#include <map>

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class Opticks ; 

class CG4 ; 
struct CG4Ctx ; 

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


class CFG4_API CRandomEngine : public CLHEP::HepRandomEngine 
{
    public:
        static std::string FormLocation();
        static std::string FormLocation(const char* file, int line);
    public:
        CRandomEngine(CG4* g4);

    protected:
        friend class CG4 ; 
        void postpropagate();
        void posttrack();
    private:
        void init(); 
        bool isNonRan() const ; 
        bool isDefault() const ; 
        void dump(const char* msg) const ; 
        void dumpCounts(const char* msg) const ; 
        void dumpDigests(const char* msg, bool locations) const ; 
        void dumpLocations(const std::vector<std::string>& digests) const ;
    public:
        std::string desc() const ; 
    public:
        std::string name() const ;
        double flat() ;  
        double flat_instrumented(const char* file, int line) ;  
        void flatArray (const int size, double* vect);
    private:
        CG4*                     m_g4 ; 
        CG4Ctx&                  m_ctx ; 
        Opticks*                 m_ok ; 
        long                     m_seed ; 
        bool                     m_internal ; 
        bool                     m_skipdupe ; 

        CLHEP::HepJamesRandom*   m_james ; 
        CLHEP::NonRandomEngine*  m_nonran ; 
        CLHEP::HepRandomEngine*  m_engine ; 

        unsigned                      m_count ; 
        unsigned                      m_count_mismatch; 
        int                           m_harikari ; 
        std::map<unsigned, unsigned>  m_record_count ; 

        std::string                   m_location ; 
        std::vector<std::string>      m_location_vec ; 
        std::string                   m_digest ;
 
        std::map<std::string, unsigned>                m_digest_count ; 
        std::map<std::string, unsigned long long>      m_digest_seqhis ; 
        std::map<std::string, std::string>             m_digest_locations ; 
        
    private:

        void setSeed(long , int) ; 
        void setSeeds(const long * , int) ; 
        void saveStatus( const char * ) const ; 
        void restoreStatus( const char * ); 
        void showStatus() const ; 




};

#include "CFG4_TAIL.hh"

