#pragma once

/**
SProcessHits_EPH.h
====================

**/


#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include "NP.hh"

namespace EPH
{
    // enumerate ProcessHits return
    enum {
       UNSET,        // 0
       NDIS,         // 1
       NOPT,         // 2
       NEDEP,        // 3
       NBOUND,       // 4
       NPROC,        // 5
       NDETECT,      // 6
       NDECULL,      // 7
       YMERGE,       // 8
       YSAVE,        // 9
       SAVENORM,     // 10
       SAVEMUON      // 11
    };

    static constexpr const char* UNSET_   = "EPH_UNSET" ;   // 0
    static constexpr const char* NDIS_    = "EPH_NDIS" ;    // 1
    static constexpr const char* NOPT_    = "EPH_NOPT" ;    // 2
    static constexpr const char* NEDEP_   = "EPH_NEDEP" ;   // 3
    static constexpr const char* NBOUND_  = "EPH_NBOUND" ;  // 4
    static constexpr const char* NPROC_   = "EPH_NPROC" ;   // 5
    static constexpr const char* NDETECT_ = "EPH_NDETECT" ; // 6
    static constexpr const char* NDECULL_ = "EPH_NDECULL" ; // 7
    static constexpr const char* YMERGE_  = "EPH_YMERGE" ;  // 8
    static constexpr const char* YSAVE_   = "EPH_YSAVE" ;   // 9
    static constexpr const char* SAVENORM_ = "EPH_SAVENORM" ; // 10
    static constexpr const char* SAVEMUON_ = "EPH_SAVEMUON" ; // 11


    static const char* Name(int eph)
    {
        const char* sn = nullptr ;
        switch(eph)
        {
            case UNSET:   sn = UNSET_    ; break ;  // 0
            case NDIS:    sn = NDIS_     ; break ;  // 1
            case NOPT:    sn = NOPT_     ; break ;  // 2
            case NEDEP:   sn = NEDEP_    ; break ;  // 3
            case NBOUND:  sn = NBOUND_   ; break ;  // 4
            case NPROC:   sn = NPROC_    ; break ;  // 5
            case NDETECT: sn = NDETECT_  ; break ;  // 6
            case NDECULL: sn = NDECULL_  ; break ;  // 7
            case YMERGE:  sn = YMERGE_   ; break ;  // 8
            case YSAVE:   sn = YSAVE_    ; break ;  // 9
            case SAVENORM: sn = SAVENORM_  ; break ;  // 10
            case SAVEMUON: sn = SAVEMUON_  ; break ;  // 11
        }
        return sn ;
    }

    static void GetNames(std::vector<std::string>& names)
    {
        names.push_back(Name(UNSET));   // 0
        names.push_back(Name(NDIS));    // 1
        names.push_back(Name(NOPT));    // 2
        names.push_back(Name(NEDEP));   // 3
        names.push_back(Name(NBOUND));  // 4
        names.push_back(Name(NPROC));   // 5
        names.push_back(Name(NDETECT)); // 6
        names.push_back(Name(NDECULL)); // 7
        names.push_back(Name(YMERGE));  // 8
        names.push_back(Name(YSAVE));   // 9
        names.push_back(Name(SAVENORM)); // 10
        names.push_back(Name(SAVEMUON)); // 11
    }
    // end of EPH namespace
}


struct SProcessHits_EPH
{
    static constexpr const char* ProcessHits_count_ = "ProcessHits_count" ;
    static constexpr const char* ProcessHits_true_  = "ProcessHits_true" ;
    static constexpr const char* ProcessHits_false_ = "ProcessHits_false" ;
    static constexpr const char* SaveNormHit_count_ = "SaveNormHit_count" ;
    static constexpr const char* SaveMuonHit_count_ = "SaveMuonHit_count" ;

    static constexpr const char* Initialize_G4HCofThisEvent_opticksMode_ = "Initialize_G4HCofThisEvent_opticksMode" ;
    static constexpr const char* Initialize_G4HCofThisEvent_count_ = "Initialize_G4HCofThisEvent_count" ;

    static constexpr const char* EndOfEvent_Simulate_eventID_ = "EndOfEvent_Simulate_eventID" ;
    static constexpr const char* EndOfEvent_Simulate_count_   = "EndOfEvent_Simulate_count" ;
    static constexpr const char* EndOfEvent_Simulate_EGPU_hit_ = "EndOfEvent_Simulate_EGPU_hit" ;
    static constexpr const char* EndOfEvent_hitCollection_entries0_ = "EndOfEvent_hitCollection_entries0" ;
    static constexpr const char* EndOfEvent_hitCollection_entries1_ = "EndOfEvent_hitCollection_entries1" ;
    static constexpr const char* EndOfEvent_hitCollectionAlt_entries0_ = "EndOfEvent_hitCollectionAlt_entries0" ;
    static constexpr const char* EndOfEvent_hitCollectionAlt_entries1_ = "EndOfEvent_hitCollectionAlt_entries1" ;
    static constexpr const char* EndOfEvent_Simulate_merged_count_   = "EndOfEvent_Simulate_merged_count" ;
    static constexpr const char* EndOfEvent_Simulate_savehit_count_   = "EndOfEvent_Simulate_savehit_count" ;
    static constexpr const char* EndOfEvent_Simulate_merged_total_   = "EndOfEvent_Simulate_merged_total" ;
    static constexpr const char* EndOfEvent_Simulate_savehit_total_   = "EndOfEvent_Simulate_savehit_total" ;


    int ProcessHits_count ;
    int ProcessHits_true  ;
    int ProcessHits_false ;
    int SaveNormHit_count ;
    int SaveMuonHit_count ;

    int UNSET ;         // 0
    int NDIS ;          // 1
    int NOPT ;          // 2
    int NEDEP ;         // 3
    int NBOUND ;        // 4
    int NPROC ;         // 5
    int NDETECT ;       // 6
    int NDECULL ;       // 7
    int YMERGE ;        // 8
    int YSAVE ;         // 9
    int SAVENORM ;       // 10
    int SAVEMUON ;       // 11


    int Initialize_G4HCofThisEvent_opticksMode ;
    int Initialize_G4HCofThisEvent_count ;

    int EndOfEvent_Simulate_eventID ;
    int EndOfEvent_Simulate_count ;
    int EndOfEvent_Simulate_EGPU_hit ;
    int EndOfEvent_hitCollection_entries0 ;
    int EndOfEvent_hitCollection_entries1 ;
    int EndOfEvent_hitCollectionAlt_entries0 ;
    int EndOfEvent_hitCollectionAlt_entries1 ;
    int EndOfEvent_Simulate_merged_count ;
    int EndOfEvent_Simulate_savehit_count ;
    int EndOfEvent_Simulate_merged_total ;
    int EndOfEvent_Simulate_savehit_total ;

    SProcessHits_EPH();
    void zero() ;
    std::string desc() const ;
    void add(int eph, bool processHits);
    void get_kvs( std::vector<std::pair<std::string, int>>& kv ) const ;

    NP* get_meta_array() const ;  // dummy array that carries metadata

};


inline SProcessHits_EPH::SProcessHits_EPH()
{
    zero();
}
inline void SProcessHits_EPH::zero()
{
    ProcessHits_count = 0 ;
    ProcessHits_true = 0 ;
    ProcessHits_false = 0 ;
    SaveNormHit_count = 0 ;
    SaveMuonHit_count = 0 ;

    UNSET = 0 ;
    NDIS = 0 ;
    NOPT = 0 ;
    NEDEP = 0 ;
    NBOUND = 0 ;
    NPROC = 0 ;
    NDETECT = 0 ;
    NDECULL = 0 ;
    YMERGE = 0 ;
    YSAVE = 0 ;
    SAVENORM = 0 ;
    SAVEMUON = 0 ;

    Initialize_G4HCofThisEvent_opticksMode = 0 ;
    Initialize_G4HCofThisEvent_count = 0 ;

    EndOfEvent_Simulate_eventID = 0 ;
    EndOfEvent_Simulate_count = 0 ;
    EndOfEvent_Simulate_EGPU_hit = 0 ;
    EndOfEvent_hitCollection_entries0 = 0 ;
    EndOfEvent_hitCollection_entries1 = 0 ;
    EndOfEvent_hitCollectionAlt_entries0 = 0 ;
    EndOfEvent_hitCollectionAlt_entries1 = 0 ;
    EndOfEvent_Simulate_merged_count = 0 ;
    EndOfEvent_Simulate_savehit_count = 0 ;
    EndOfEvent_Simulate_merged_total = 0 ;
    EndOfEvent_Simulate_savehit_total = 0 ;
}

inline std::string SProcessHits_EPH::desc() const
{
    typedef std::pair<std::string, int> SI ;
    typedef std::vector<SI> VSI ;
    VSI kvs ;
    get_kvs(kvs);
    int w = 50 ;
    std::stringstream ss ;
    ss << "[SProcessHits_EPH::desc\n" ;
    for(int i=0 ; i < int(kvs.size()) ; i++) ss << std::setw(w) << kvs[i].first << std::setw(8) << kvs[i].second << "\n" ;
    ss << "]SProcessHits_EPH::desc\n" ;
    std::string str = ss.str();
    return str ;
}


inline void SProcessHits_EPH::add(int eph, bool processHits)
{
    ProcessHits_count  += 1 ;
    ProcessHits_true   += int(processHits) ;
    ProcessHits_false  += int(!processHits) ;

    switch(eph)
    {
        case EPH::UNSET:   UNSET   += 1 ; break ;  // 0
        case EPH::NDIS:    NDIS    += 1 ; break ;  // 1
        case EPH::NOPT:    NOPT    += 1 ; break ;  // 2
        case EPH::NEDEP:   NEDEP   += 1 ; break ;  // 3
        case EPH::NBOUND:  NBOUND  += 1 ; break ;  // 4
        case EPH::NPROC:   NPROC   += 1 ; break ;  // 5
        case EPH::NDETECT: NDETECT += 1 ; break ;  // 6
        case EPH::NDECULL: NDECULL += 1 ; break ;  // 7
        case EPH::YMERGE:  YMERGE  += 1 ; break ;  // 8
        case EPH::YSAVE:   YSAVE   += 1 ; break ;  // 9
        case EPH::SAVENORM: SAVENORM += 1 ; break ;  // 10
        case EPH::SAVEMUON: SAVEMUON += 1 ; break ;  // 11
    }
}




inline void SProcessHits_EPH::get_kvs( std::vector<std::pair<std::string, int>>& kvs ) const
{
    typedef std::pair<std::string, int> KV ;
    kvs.push_back(KV(ProcessHits_count_, ProcessHits_count));
    kvs.push_back(KV(ProcessHits_true_, ProcessHits_true));
    kvs.push_back(KV(ProcessHits_false_, ProcessHits_false));
    kvs.push_back(KV(SaveNormHit_count_, SaveNormHit_count));
    kvs.push_back(KV(SaveMuonHit_count_, SaveMuonHit_count));

    kvs.push_back(KV(EPH::UNSET_, UNSET));
    kvs.push_back(KV(EPH::NDIS_, NDIS));
    kvs.push_back(KV(EPH::NOPT_, NOPT));
    kvs.push_back(KV(EPH::NEDEP_, NEDEP));
    kvs.push_back(KV(EPH::NBOUND_, NBOUND));
    kvs.push_back(KV(EPH::NPROC_, NPROC));
    kvs.push_back(KV(EPH::NDETECT_, NDETECT));
    kvs.push_back(KV(EPH::NDECULL_, NDECULL));
    kvs.push_back(KV(EPH::YMERGE_, YMERGE));
    kvs.push_back(KV(EPH::YSAVE_, YSAVE));
    kvs.push_back(KV(EPH::SAVENORM_, SAVENORM));
    kvs.push_back(KV(EPH::SAVEMUON_, SAVEMUON));

    kvs.push_back(KV(Initialize_G4HCofThisEvent_opticksMode_, Initialize_G4HCofThisEvent_opticksMode));
    kvs.push_back(KV(Initialize_G4HCofThisEvent_count_, Initialize_G4HCofThisEvent_count));

    kvs.push_back(KV(EndOfEvent_Simulate_eventID_,    EndOfEvent_Simulate_eventID));
    kvs.push_back(KV(EndOfEvent_Simulate_count_,    EndOfEvent_Simulate_count));
    kvs.push_back(KV(EndOfEvent_Simulate_EGPU_hit_, EndOfEvent_Simulate_EGPU_hit));
    kvs.push_back(KV(EndOfEvent_hitCollection_entries0_, EndOfEvent_hitCollection_entries0));
    kvs.push_back(KV(EndOfEvent_hitCollection_entries1_, EndOfEvent_hitCollection_entries1));
    kvs.push_back(KV(EndOfEvent_hitCollectionAlt_entries0_, EndOfEvent_hitCollectionAlt_entries0));
    kvs.push_back(KV(EndOfEvent_hitCollectionAlt_entries1_, EndOfEvent_hitCollectionAlt_entries1));
    kvs.push_back(KV(EndOfEvent_Simulate_merged_count_,  EndOfEvent_Simulate_merged_count));
    kvs.push_back(KV(EndOfEvent_Simulate_savehit_count_, EndOfEvent_Simulate_savehit_count));
    kvs.push_back(KV(EndOfEvent_Simulate_merged_total_,  EndOfEvent_Simulate_merged_total));
    kvs.push_back(KV(EndOfEvent_Simulate_savehit_total_, EndOfEvent_Simulate_savehit_total));
}


inline NP* SProcessHits_EPH::get_meta_array() const
{
    NP* meta = NP::Make<int>(1) ;
    EPH::GetNames(meta->names) ;

    std::vector<std::pair<std::string, int>> kvs ;
    get_kvs(kvs);
    meta->set_meta_kv(kvs);
    return meta ;
}


