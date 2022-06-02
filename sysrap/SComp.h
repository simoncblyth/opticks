#pragma once
/**
SComp.h : naming array components of QEvent/SEvt
=============================================================

NB: the old OpticksEvent analog of this is SComponent.hh

**/

#include <string>
#include <vector>
#include <sstream>
#include <cstring>
#include "SYSRAP_API_EXPORT.hh"

struct NP ; 

enum {
    SCOMP_UNDEFINED = 0x1 <<  0, 
    SCOMP_GENSTEP   = 0x1 <<  1, 
    SCOMP_PHOTON    = 0x1 <<  2,
    SCOMP_RECORD    = 0x1 <<  3, 
    SCOMP_REC       = 0x1 <<  4,
    SCOMP_SEED      = 0x1 <<  5,
    SCOMP_HIT       = 0x1 <<  6,
    SCOMP_SIMTRACE  = 0x1 <<  7,
    SCOMP_SEQ       = 0x1 <<  8,
    SCOMP_DOMAIN    = 0x1 <<  9
};

struct SYSRAP_API SCompProvider
{
    virtual NP* getComponent(unsigned comp) const = 0 ; 
}; 

struct SYSRAP_API SComp
{
    static constexpr const char* ALL_ = "genstep,photon,record,rec,seq,seed,hit,simtrace,domain" ; 
    static constexpr const char* UNDEFINED_ = "undefined" ; 
    static constexpr const char* GENSTEP_   = "genstep" ; 
    static constexpr const char* PHOTON_    = "photon" ; 
    static constexpr const char* RECORD_    = "record" ; 
    static constexpr const char* REC_       = "rec" ; 
    static constexpr const char* SEQ_       = "seq" ; 
    static constexpr const char* SEED_      = "seed" ; 
    static constexpr const char* HIT_       = "hit" ; 
    static constexpr const char* SIMTRACE_  = "simtrace" ; 
    static constexpr const char* DOMAIN_    = "domain" ; 

    static bool Match(const char* q, const char* n ); 
    static unsigned    Comp(const char* name); 
    static const char* Name(unsigned comp); 
    static std::string Desc(unsigned mask); 
    static unsigned    Mask(const char* names, char delim=','); 

    static bool IsGenstep( unsigned mask){ return mask & SCOMP_GENSTEP ; }
    static bool IsPhoton(  unsigned mask){ return mask & SCOMP_PHOTON ; }
    static bool IsRecord(  unsigned mask){ return mask & SCOMP_RECORD ; }
    static bool IsRec(     unsigned mask){ return mask & SCOMP_REC ; }
    static bool IsSeed(    unsigned mask){ return mask & SCOMP_SEED ; }
    static bool IsHit(     unsigned mask){ return mask & SCOMP_HIT ; }
    static bool IsSimtrace(unsigned mask){ return mask & SCOMP_SIMTRACE ; }
    static bool IsSeq(     unsigned mask){ return mask & SCOMP_SEQ ; }
    static bool IsDomain(  unsigned mask){ return mask & SCOMP_DOMAIN ; }
};

inline bool SComp::Match(const char* q, const char* n )
{
    return q && strcmp( q, n ) == 0 ; 
}

inline unsigned SComp::Comp(const char* name)
{
    unsigned comp = SCOMP_UNDEFINED ; 
    if(Match(name, GENSTEP_))  comp = SCOMP_GENSTEP ; 
    if(Match(name, PHOTON_))   comp = SCOMP_PHOTON ; 
    if(Match(name, RECORD_))   comp = SCOMP_RECORD ; 
    if(Match(name, REC_))      comp = SCOMP_REC ; 
    if(Match(name, SEQ_))      comp = SCOMP_SEQ ; 
    if(Match(name, SEED_))     comp = SCOMP_SEED ; 
    if(Match(name, HIT_))      comp = SCOMP_HIT ; 
    if(Match(name, SIMTRACE_)) comp = SCOMP_SIMTRACE ; 
    if(Match(name, DOMAIN_))   comp = SCOMP_DOMAIN ; 
    return comp ; 
}
inline const char* SComp::Name(unsigned comp)
{
    const char* s = nullptr ; 
    switch(comp)
    {
        case SCOMP_UNDEFINED: s = UNDEFINED_  ; break ;  
        case SCOMP_GENSTEP:   s = GENSTEP_    ; break ;  
        case SCOMP_PHOTON:    s = PHOTON_     ; break ;  
        case SCOMP_RECORD:    s = RECORD_     ; break ;  
        case SCOMP_REC:       s = REC_        ; break ;  
        case SCOMP_SEQ:       s = SEQ_        ; break ;  
        case SCOMP_SEED:      s = SEED_       ; break ;  
        case SCOMP_HIT:       s = HIT_        ; break ;  
        case SCOMP_SIMTRACE:  s = SIMTRACE_   ; break ;  
        case SCOMP_DOMAIN:    s = DOMAIN_     ; break ;  
    }
    return s ; 
}
inline std::string SComp::Desc(unsigned mask)
{
    // curious using vector of const char* gives undefined symbol link errors
    std::vector<std::string> names ;   
    if( mask & SCOMP_GENSTEP )  names.push_back(GENSTEP_) ;   
    if( mask & SCOMP_PHOTON )   names.push_back(PHOTON_) ; 
    if( mask & SCOMP_RECORD )   names.push_back(RECORD_) ; 
    if( mask & SCOMP_REC )      names.push_back(REC_)  ; 
    if( mask & SCOMP_SEQ )      names.push_back(SEQ_) ;  
    if( mask & SCOMP_SEED )     names.push_back(SEED_) ; 
    if( mask & SCOMP_HIT )      names.push_back(HIT_) ; 
    if( mask & SCOMP_SIMTRACE ) names.push_back(SIMTRACE_) ; 
    if( mask & SCOMP_DOMAIN )   names.push_back(DOMAIN_) ; 

    std::stringstream ss ; 
    for(unsigned i=0 ; i < names.size() ; i++) ss << names[i] << ( i < names.size() - 1 ? "," : "" ); 
    std::string s = ss.str(); 
    return s ; 
}
inline unsigned SComp::Mask(const char* names, char delim)
{
    unsigned mask = 0 ; 
    std::stringstream ss;  
    ss.str(names)  ;
    std::string s;
    while (std::getline(ss, s, delim)) mask |= Comp(s.c_str())  ; 
    return mask ; 
}

