#pragma once
/**
SComp.h : naming array components of QEvt/SEvt
=============================================================

NB: the old OpticksEvent analog of this is SComponent.hh

**/

#include <string>
#include <vector>
#include <sstream>
#include <cstring>
#include <bitset>
#include "SYSRAP_API_EXPORT.hh"

struct NP ;

enum {
    SCOMP_UNDEFINED    = 0x1 <<  0,
    SCOMP_GENSTEP      = 0x1 <<  1,
    SCOMP_PHOTON       = 0x1 <<  2,
    SCOMP_RECORD       = 0x1 <<  3,
    SCOMP_REC          = 0x1 <<  4,
    SCOMP_SEQ          = 0x1 <<  5,
    SCOMP_PRD          = 0x1 <<  6,
    SCOMP_SEED         = 0x1 <<  7,
    SCOMP_HIT          = 0x1 <<  8,
    SCOMP_SIMTRACE     = 0x1 <<  9,
    SCOMP_DOMAIN       = 0x1 << 10,
    SCOMP_INPHOTON     = 0x1 << 11,
    SCOMP_TAG          = 0x1 << 12,
    SCOMP_FLAT         = 0x1 << 13,
    SCOMP_ISECT        = 0x1 << 14,
    SCOMP_FPHOTON      = 0x1 << 15,
    SCOMP_PIXEL        = 0x1 << 16,
    SCOMP_G4STATE      = 0x1 << 17,
    SCOMP_AUX          = 0x1 << 18,
    SCOMP_SUP          = 0x1 << 19,
    SCOMP_PHO          = 0x1 << 20,
    SCOMP_GS           = 0x1 << 21,
    SCOMP_PHOTONLITE   = 0x1 << 22,
    SCOMP_HITLITE      = 0x1 << 23,
    SCOMP_HITLOCAL     = 0x1 << 24,
    SCOMP_PHOTONLOCAL  = 0x1 << 25

};

struct SYSRAP_API SCompProvider
{
    virtual const char* getTypeName() const = 0 ;
    virtual std::string getMeta() const = 0 ;
    virtual NP* gatherComponent(unsigned comp) const = 0 ;
};

struct SYSRAP_API SComp
{
    static constexpr const char* NONE_ = "" ;
    static constexpr const char* ALL_ = "genstep,photon,record,rec,seq,prd,seed,hit,simtrace,domain,inphoton,tag,flat" ;
    static constexpr const char* UNDEFINED_ = "undefined" ;

    static constexpr const char* GENSTEP_   = "genstep" ;
    static constexpr const char* PHOTON_    = "photon" ;
    static constexpr const char* RECORD_    = "record" ;
    static constexpr const char* REC_       = "rec" ;
    static constexpr const char* SEQ_       = "seq" ;
    static constexpr const char* PRD_       = "prd" ;
    static constexpr const char* SEED_      = "seed" ;
    static constexpr const char* HIT_       = "hit" ;
    static constexpr const char* SIMTRACE_  = "simtrace" ;
    static constexpr const char* DOMAIN_    = "domain" ;
    static constexpr const char* INPHOTON_  = "inphoton" ;
    static constexpr const char* TAG_       = "tag" ;
    static constexpr const char* FLAT_      = "flat" ;

    static constexpr const char* ISECT_     = "isect" ;
    static constexpr const char* FPHOTON_   = "fphoton" ;
    static constexpr const char* PIXEL_     = "pixel" ;
    static constexpr const char* G4STATE_   = "g4state" ;
    static constexpr const char* AUX_       = "aux" ;
    static constexpr const char* SUP_       = "sup" ;
    static constexpr const char* PHO_       = "pho" ;
    static constexpr const char* GS_        = "gs" ;
    static constexpr const char* PHOTONLITE_ = "photonlite" ;
    static constexpr const char* HITLITE_    = "hitlite" ;
    static constexpr const char* HITLOCAL_   = "hitlocal" ;
    static constexpr const char* PHOTONLOCAL_ = "photonlocal" ;

    static bool Match(const char* q, const char* n );
    static unsigned    Comp(const char* name);
    static const char* Name(unsigned comp);
    static std::string Desc(unsigned mask);
    static std::string Desc(const std::vector<unsigned>& comps);
    static void CompList(std::vector<unsigned>& comps, const char* names, char delim=',');
    static void CompListAll( std::vector<unsigned>& comps );
    static void CompListMask(std::vector<unsigned>& comps, unsigned mask );
    static int  CompListCount(unsigned mask );
    static unsigned    Mask(const char* names, char delim=',');


    static bool IsGenstep( unsigned mask){ return mask & SCOMP_GENSTEP ; }
    static bool IsPhoton(  unsigned mask){ return mask & SCOMP_PHOTON ; }
    static bool IsRecord(  unsigned mask){ return mask & SCOMP_RECORD ; }
    static bool IsRec(     unsigned mask){ return mask & SCOMP_REC ; }
    static bool IsSeq(     unsigned mask){ return mask & SCOMP_SEQ ; }
    static bool IsPrd(     unsigned mask){ return mask & SCOMP_PRD ; }
    static bool IsSeed(    unsigned mask){ return mask & SCOMP_SEED ; }
    static bool IsHit(     unsigned mask){ return mask & SCOMP_HIT ; }
    static bool IsSimtrace(unsigned mask){ return mask & SCOMP_SIMTRACE ; }
    static bool IsDomain(  unsigned mask){ return mask & SCOMP_DOMAIN ; }
    static bool IsInphoton(unsigned mask){ return mask & SCOMP_INPHOTON ; }
    static bool IsTag(     unsigned mask){ return mask & SCOMP_TAG ; }
    static bool IsFlat(    unsigned mask){ return mask & SCOMP_FLAT ; }
    static bool IsIsect(   unsigned mask){ return mask & SCOMP_ISECT ; }
    static bool IsFphoton( unsigned mask){ return mask & SCOMP_FPHOTON ; }
    static bool IsPixel(   unsigned mask){ return mask & SCOMP_PIXEL ; }
    static bool IsG4State( unsigned mask){ return mask & SCOMP_G4STATE ; }
    static bool IsAux(     unsigned mask){ return mask & SCOMP_AUX ; }
    static bool IsSup(     unsigned mask){ return mask & SCOMP_SUP ; }
    static bool IsPho(     unsigned mask){ return mask & SCOMP_PHO ; }
    static bool IsGS(      unsigned mask){ return mask & SCOMP_GS ; }
    static bool IsPhotonLite(  unsigned mask){ return mask & SCOMP_PHOTONLITE ; }
    static bool IsHitLite(     unsigned mask){ return mask & SCOMP_HITLITE ; }
    static bool IsHitLocal(    unsigned mask){ return mask & SCOMP_HITLOCAL ; }
    static bool IsPhotonLocal( unsigned mask){ return mask & SCOMP_PHOTONLOCAL ; }

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
    if(Match(name, PRD_))      comp = SCOMP_PRD ;
    if(Match(name, SEED_))     comp = SCOMP_SEED ;
    if(Match(name, HIT_))      comp = SCOMP_HIT ;
    if(Match(name, SIMTRACE_)) comp = SCOMP_SIMTRACE ;
    if(Match(name, DOMAIN_))   comp = SCOMP_DOMAIN ;
    if(Match(name, INPHOTON_)) comp = SCOMP_INPHOTON ;
    if(Match(name, TAG_))      comp = SCOMP_TAG ;
    if(Match(name, FLAT_))     comp = SCOMP_FLAT ;
    if(Match(name, ISECT_))    comp = SCOMP_ISECT ;
    if(Match(name, FPHOTON_))  comp = SCOMP_FPHOTON ;
    if(Match(name, PIXEL_))    comp = SCOMP_PIXEL ;
    if(Match(name, G4STATE_))  comp = SCOMP_G4STATE ;
    if(Match(name, AUX_))      comp = SCOMP_AUX ;
    if(Match(name, SUP_))      comp = SCOMP_SUP ;
    if(Match(name, PHO_))      comp = SCOMP_PHO ;
    if(Match(name, GS_))       comp = SCOMP_GS ;
    if(Match(name, PHOTONLITE_)) comp = SCOMP_PHOTONLITE ;
    if(Match(name, HITLITE_))    comp = SCOMP_HITLITE ;
    if(Match(name, HITLOCAL_))   comp = SCOMP_HITLOCAL ;
    if(Match(name, PHOTONLOCAL_))   comp = SCOMP_PHOTONLOCAL ;
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
        case SCOMP_PRD:       s = PRD_        ; break ;
        case SCOMP_SEED:      s = SEED_       ; break ;
        case SCOMP_HIT:       s = HIT_        ; break ;
        case SCOMP_SIMTRACE:  s = SIMTRACE_   ; break ;
        case SCOMP_DOMAIN:    s = DOMAIN_     ; break ;
        case SCOMP_INPHOTON:  s = INPHOTON_   ; break ;
        case SCOMP_TAG:       s = TAG_        ; break ;
        case SCOMP_FLAT:      s = FLAT_       ; break ;
        case SCOMP_ISECT:     s = ISECT_      ; break ;
        case SCOMP_FPHOTON:   s = FPHOTON_    ; break ;
        case SCOMP_PIXEL:     s = PIXEL_      ; break ;
        case SCOMP_G4STATE:   s = G4STATE_    ; break ;
        case SCOMP_AUX:       s = AUX_        ; break ;
        case SCOMP_SUP:       s = SUP_        ; break ;
        case SCOMP_PHO:       s = PHO_        ; break ;
        case SCOMP_GS:        s = GS_         ; break ;
        case SCOMP_PHOTONLITE:s = PHOTONLITE_ ; break ;
        case SCOMP_HITLITE:   s = HITLITE_    ; break ;
        case SCOMP_HITLOCAL:  s = HITLOCAL_   ; break ;
        case SCOMP_PHOTONLOCAL:  s = PHOTONLOCAL_   ; break ;
    }
    return s ;
}
inline std::string SComp::Desc(unsigned mask)
{
    // curious using vector of const char* gives undefined symbol link errors : something funny with "static constexpr const char*" ?
    std::vector<std::string> names ;
    if( mask & SCOMP_GENSTEP )  names.push_back(GENSTEP_) ;
    if( mask & SCOMP_PHOTON )   names.push_back(PHOTON_) ;
    if( mask & SCOMP_RECORD )   names.push_back(RECORD_) ;
    if( mask & SCOMP_REC )      names.push_back(REC_)  ;
    if( mask & SCOMP_SEQ )      names.push_back(SEQ_) ;
    if( mask & SCOMP_PRD )      names.push_back(PRD_) ;
    if( mask & SCOMP_SEED )     names.push_back(SEED_) ;
    if( mask & SCOMP_HIT )      names.push_back(HIT_) ;   // CAUTION : HIT MUST STAY AFTER PHOTON IN THE NAMES
    if( mask & SCOMP_SIMTRACE ) names.push_back(SIMTRACE_) ;
    if( mask & SCOMP_DOMAIN )   names.push_back(DOMAIN_) ;
    if( mask & SCOMP_INPHOTON ) names.push_back(INPHOTON_) ;
    if( mask & SCOMP_TAG )      names.push_back(TAG_) ;
    if( mask & SCOMP_FLAT )     names.push_back(FLAT_) ;
    if( mask & SCOMP_ISECT )    names.push_back(ISECT_) ;
    if( mask & SCOMP_FPHOTON )  names.push_back(FPHOTON_) ;
    if( mask & SCOMP_PIXEL )    names.push_back(PIXEL_) ;
    if( mask & SCOMP_G4STATE )  names.push_back(G4STATE_) ;
    if( mask & SCOMP_AUX )      names.push_back(AUX_) ;
    if( mask & SCOMP_SUP )      names.push_back(SUP_) ;
    if( mask & SCOMP_PHO )      names.push_back(PHO_) ;
    if( mask & SCOMP_GS )       names.push_back(GS_) ;
    if( mask & SCOMP_PHOTONLITE ) names.push_back(PHOTONLITE_) ;
    if( mask & SCOMP_HITLITE )    names.push_back(HITLITE_) ;   // CAUTION : HITLITE MUST STAY AFTER PHOTONLITE (?)
    if( mask & SCOMP_HITLOCAL )   names.push_back(HITLOCAL_) ;   // CAUTION : HITLOCAL MUST STAY AFTER HIT
    if( mask & SCOMP_PHOTONLOCAL )   names.push_back(PHOTONLOCAL_) ;   // CAUTION : PHOTONLOCAL MUST STAY AFTER PHOTON

    std::stringstream ss ;
    for(unsigned i=0 ; i < names.size() ; i++) ss << names[i] << ( i < names.size() - 1 ? "," : "" );
    std::string s = ss.str();
    return s ;
}
inline std::string SComp::Desc(const std::vector<unsigned>& comps)
{
    std::stringstream ss ;
    for(unsigned i=0 ; i < comps.size() ; i++) ss << Name(comps[i]) << ( i < comps.size() - 1 ? "," : "" );
    std::string s = ss.str();
    return s ;
}


/**
SComp::CompList
----------------

1. Split *names* using the *delim*
2. collect into *comps* vector the unsigned int corresponding to each name

**/

inline void SComp::CompList(std::vector<unsigned>& comps, const char* names, char delim )
{
    if(!names) return ;
    std::stringstream ss;
    ss.str(names)  ;
    std::string s;
    while (std::getline(ss, s, delim)) comps.push_back( Comp(s.c_str()) ) ;
}

inline void SComp::CompListAll(std::vector<unsigned>& comps )
{
    CompList(comps, ALL_, ',' );
}

inline void SComp::CompListMask(std::vector<unsigned>& comps, unsigned mask )
{
    std::bitset<32> msk(mask);
    for(unsigned i=0 ; i < msk.size() ; i++) if(msk[i]) comps.push_back( 0x1 << i ) ;
}
inline int SComp::CompListCount(unsigned mask )
{
    std::vector<unsigned> comps;
    CompListMask(comps, mask);
    return comps.size() ;
}


/**
SComp::Mask
-------------

1. Use *CompList* to collect into *comps* vector the unsigned int corresponding to each name from *delim* *names*
2. return the bitwise-OR of all the comp integers

**/


inline unsigned SComp::Mask(const char* names, char delim)
{
    std::vector<unsigned> comps ;
    CompList(comps, names, delim );
    unsigned mask = 0 ;
    for(unsigned i=0 ; i < comps.size() ; i++)  mask |= comps[i] ;
    return mask ;
}




