#pragma once
/**
SComponent.hh : naming array components of Opticks Event 
=============================================================

NB new workflow uses SComp.h NOT THIS 


Pulling out parts of okc/OpticksEvent to allow development of less repetitive sysrap/SOpticksEvent

* NP metadata capabilities can probably eliminate FDOM, IDOM ?

**/

#include <cstring>

enum {
    SEVT_UNDEFINED, 
    SEVT_GENSTEP, 
    SEVT_NOPSTEP,
    SEVT_PHOTON,
    SEVT_DEBUG,
    SEVT_WAY, 
    SEVT_SOURCE,
    SEVT_RECORD,
    SEVT_DELUXE,
    SEVT_PHOSEL,
    SEVT_RECSEL,
    SEVT_SEQUENCE,
    SEVT_BOUNDARY,
    SEVT_SEED, 
    SEVT_HIT,
    SEVT_HIY,
    SEVT_FDOM,
    SEVT_IDOM
};

struct SComponent 
{
    static bool        StartsWith( const char* s, const char* q); 
    static const char* Name(unsigned comp); 
    static unsigned    Component(const char* name); 

    static constexpr const char* UNDEFINED_ = "undefined" ; 
    static constexpr const char* GENSTEP_ = "genstep" ; 
    static constexpr const char* NOPSTEP_ = "nopstep" ; 
    static constexpr const char* PHOTON_  = "photon" ; 
    static constexpr const char* DEBUG_   = "debug" ; 
    static constexpr const char* WAY_     = "way" ; 
    static constexpr const char* SOURCE_  = "source" ; 
    static constexpr const char* RECORD_  = "record" ; 
    static constexpr const char* DELUXE_  = "deluxe" ; 
    static constexpr const char* PHOSEL_  = "phosel" ; 
    static constexpr const char* RECSEL_  = "recsel" ; 
    static constexpr const char* SEQUENCE_ = "sequence" ; 
    static constexpr const char* BOUNDARY_ = "boundary" ; 
    static constexpr const char* SEED_     = "seed" ; 
    static constexpr const char* HIT_      = "hit" ; 
    static constexpr const char* HIY_      = "hiy" ; 
    static constexpr const char* FDOM_     = "fdom" ; 
    static constexpr const char* IDOM_     = "idom" ; 
};

inline bool SComponent::StartsWith( const char* s, const char* q)  // static
{
    return s && q && strlen(q) <= strlen(s) && strncmp(s, q, strlen(q)) == 0 ; 
}

inline unsigned    SComponent::Component(const char* name)
{
    unsigned comp = SEVT_UNDEFINED ; 
    if(StartsWith(name, GENSTEP_))  comp = SEVT_GENSTEP ; 
    if(StartsWith(name, NOPSTEP_))  comp = SEVT_NOPSTEP ; 
    if(StartsWith(name, PHOTON_))   comp = SEVT_PHOTON ; 
    if(StartsWith(name, DEBUG_))    comp = SEVT_DEBUG ; 
    if(StartsWith(name, WAY_))      comp = SEVT_WAY ; 
    if(StartsWith(name, SOURCE_))   comp = SEVT_SOURCE ; 
    if(StartsWith(name, RECORD_))   comp = SEVT_RECORD ; 
    if(StartsWith(name, DELUXE_))   comp = SEVT_DELUXE ; 
    if(StartsWith(name, PHOSEL_))   comp = SEVT_PHOSEL ; 
    if(StartsWith(name, RECSEL_))   comp = SEVT_RECSEL ; 
    if(StartsWith(name, SEQUENCE_)) comp = SEVT_SEQUENCE ; 
    if(StartsWith(name, BOUNDARY_)) comp = SEVT_BOUNDARY ; 
    if(StartsWith(name, SEED_))     comp = SEVT_SEED ; 
    if(StartsWith(name, HIT_))      comp = SEVT_HIT ; 
    if(StartsWith(name, HIY_))      comp = SEVT_HIY ; 
    if(StartsWith(name, FDOM_))     comp = SEVT_FDOM ; 
    if(StartsWith(name, IDOM_))     comp = SEVT_IDOM ; 
    return comp ; 
}

inline const char* SComponent::Name(unsigned comp)
{
    const char* s = nullptr ; 
    switch(comp)
    {
        case SEVT_UNDEFINED: s = UNDEFINED_  ; break ;  
        case SEVT_GENSTEP:   s = GENSTEP_  ; break ;  
        case SEVT_NOPSTEP:   s = NOPSTEP_  ; break ;  
        case SEVT_PHOTON:    s = PHOTON_   ; break ;  
        case SEVT_DEBUG:     s = DEBUG_    ; break ;  
        case SEVT_WAY:       s = WAY_      ; break ;  
        case SEVT_SOURCE:    s = SOURCE_   ; break ;  
        case SEVT_RECORD:    s = RECORD_   ; break ;  
        case SEVT_DELUXE:    s = DELUXE_   ; break ;  
        case SEVT_PHOSEL:    s = PHOSEL_   ; break ;  
        case SEVT_RECSEL:    s = RECSEL_   ; break ;  
        case SEVT_SEQUENCE:  s = SEQUENCE_ ; break ;  
        case SEVT_BOUNDARY:  s = BOUNDARY_ ; break ;  
        case SEVT_SEED:      s = SEED_     ; break ;  
        case SEVT_HIT:       s = HIT_      ; break ;  
        case SEVT_HIY:       s = HIY_      ; break ;  
        case SEVT_FDOM:      s = FDOM_     ; break ;  
        case SEVT_IDOM:      s = IDOM_     ; break ;  
    }
    return s ; 
}

