#pragma once

#include <map>
#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class Opticks ; 

class CG4 ; 
struct CG4Ctx ; 

/**
CRandomEngine
=====

Canonical m_rng instance is resident of CG4 and is instanciated with it, 
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
        CRandomEngine(CG4* g4);

    protected:
        friend class CG4 ; 
        void postpropagate();
    private:
        void init(); 
        bool isNonRan() const ; 
        bool isDefault() const ; 
        void dump(const char* msg) const ; 
    public:
        std::string name() const ;
        double flat() ;  
        void flatArray (const int size, double* vect);
    private:
        CG4*                     m_g4 ; 
        CG4Ctx&                  m_ctx ; 
        Opticks*                 m_ok ; 
        long                     m_seed ; 

        CLHEP::HepJamesRandom*   m_james ; 
        CLHEP::NonRandomEngine*  m_nonran ; 
        CLHEP::HepRandomEngine*  m_engine ; 

        unsigned                      m_count ; 
        int                           m_harikari ; 
        std::map<unsigned, unsigned>  m_record_count ; 

        

    private:

        void setSeed(long , int) ; 
        void setSeeds(const long * , int) ; 
        void saveStatus( const char * ) const ; 
        void restoreStatus( const char * ); 
        void showStatus() const ; 




};

#include "CFG4_TAIL.hh"

