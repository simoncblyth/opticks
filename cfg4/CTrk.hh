#pragma once

#include <string>
#include "CFG4_API_EXPORT.hh"

struct CFG4_API CTrk 
{
    public:
        static unsigned    Photon_id(unsigned packed)  ;
        static char        Gentype(unsigned packed)  ;
        static bool        Reemission(unsigned packed) ;
        static std::string Desc(unsigned packed) ; 
    public:
        CTrk( const CTrk& other);
        CTrk( unsigned packed );
        CTrk( unsigned photon_id_ , char gentype_, bool reemission_ );
    public:
        unsigned    packed()     const ; 
        unsigned    photon_id()  const ; 
        char        gentype()    const ; 
        bool        reemission() const ; 
        std::string desc()       const ;  
    private:
        unsigned m_packed  ;   

};



