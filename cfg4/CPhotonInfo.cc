#include "PLOG.hh"
#include "G4Track.hh"
#include "CTrack.hh"
#include "CPhotonInfo.hh"

const plog::Severity CPhotonInfo::LEVEL = PLOG::EnvLevel("CPhotonInfo", "DEBUG") ; 

CPhotonInfo::CPhotonInfo(const CGenstep& gs, unsigned ix_, unsigned gn_, int id_ )
    :   
    G4VUserTrackInformation("CPhotonInfo"),
    pho(gs.index, ix_, id_ < 0 ? gs.offset + ix_ : id_ , gn_ )
{   
    LOG(LEVEL) ; 
}   

CPhotonInfo::CPhotonInfo(const CPho& _pho )
    :   
    G4VUserTrackInformation("CPhotonInfo"),
    pho(_pho.gs, _pho.ix, _pho.id , _pho.gn )
{
}


CPhotonInfo::~CPhotonInfo(){}
G4String* CPhotonInfo::type() const {   return pType ; }
std::string CPhotonInfo::desc() const { return pho.desc(); }

unsigned CPhotonInfo::gs() const { return pho.gs ; } // 0-based genstep index within the event
unsigned CPhotonInfo::ix() const { return pho.ix ; } // 0-based photon index within the genstep
unsigned CPhotonInfo::id() const { return pho.id ; } // 0-based absolute photon identity index within the event 
unsigned CPhotonInfo::gn() const { return pho.gn ; } // 0-based generation index incremented at each reemission  


/**
CPhotonInfo::Get
------------------

As S+C photon tracks should always be labelled 
the CPho::FabricateTrackIdPhoton should only be relevant for 
input "primary" photons. The input photons are artificially contructed 
torch 'T' photons used for debugging.

**/

CPho CPhotonInfo::Get(const G4Track* optical_photon_track, bool fabricate_unlabelled)   // static 
{
    G4VUserTrackInformation* ui = optical_photon_track->GetUserInformation() ;
    CPhotonInfo* cpui = ui ? dynamic_cast<CPhotonInfo*>(ui) : nullptr ; 
    unsigned track_id = CTrack::Id(optical_photon_track) ; 
    CPho pho ; 

    if(cpui) 
    {
        pho = cpui->pho ; 
    }
    else
    {
        if(fabricate_unlabelled)
        {
            pho = CPho::FabricateTrackIdPhoton(track_id) ;  
        }
    }
    return pho  ; 
}

/**
CPhotonInfo::MakeScintillation
----------------------------------

ancestor_id:-1 
    normal case, meaning that aTrack was not a photon
    so the generation loop will yield "primary" photons with 

    gs : genstep index
    ix : i, genstep loop index 
    id : gs.offset + i  
    gn : 0   

ancestor_id>-1
    aTrack is a photon, and ancestor_id is the absolute id of the 
    primary parent photon, this id is retained thru any subsequent 
    remission secondary generations, so

    gs : ancestor.gs
    ix : ancestor.ix
    id : ancestor.id
    gn : ancestor.gn + 1 

**/

CPhotonInfo* CPhotonInfo::MakeScintillation(const CGenstep& gs, unsigned i, const CPho& ancestor )  // static 
{
    int ancestor_id = ancestor.get_id() ;
    CPhotonInfo* spi = nullptr ; 
    if( ancestor_id == -1 ) 
    {    
        unsigned gn_ = 0 ;   // 0: primary photon generation 
        int      id_ = -1 ;  // -> use default id : gs.offset + i 
        spi = new CPhotonInfo(gs, i, gn_, id_ );    
    }
    else 
    {
        CPho reemit = ancestor.make_reemit();   // gn: generation is incremented
        spi = new CPhotonInfo(reemit); 
    }    
    return spi ; 
}

/**
CPhotonInfo::MakeCerenkov
----------------------------

gs : genstep index
ix : i, genstep loop index 
id : gs.offset + i  
gn : 0   

**/

CPhotonInfo* CPhotonInfo::MakeCerenkov( const CGenstep& gs, unsigned i ) // static
{
     unsigned gn_ = 0  ;  // generation
     int id_      = -1 ;  // -1 -> default id : gs.offset + i  
     CPhotonInfo* cpi = new CPhotonInfo(gs, i, gn_, id_ );   
     return cpi ; 
}

