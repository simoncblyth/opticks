#pragma once

/**
SProcessHits_EPH.h
====================

~/o/sysrap/tests/SProcessHits_EPH_test.sh

**/


#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cstdint>
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

    static constexpr const char* SUM_merged_count_savehit_count_   = "SUM_merged_count_savehit_count" ;
    static constexpr const char* SUM_merged_total_savehit_total_   = "SUM_merged_total_savehit_total" ;


    static constexpr const char* NOTES = R"LITERAL(
SProcessHits_EPH::NOTES
------------------------

ProcessHits_count
ProcessHits_true
ProcessHits_false
    number of calls to *SProcessHits_EPH::add* and sub-counts where is_hit is true/false,
    done from junoSD_PMT_v2::ProcessHits

SaveNormHit_count
SaveMuonHit_count
    counts collected from junoSD_PMT_v2::SaveNormHit junoSD_PMT_v2::SaveMuonHit

All 7 EndOfEvent_Simulate values are collected from junoSD_PMT_v2_Opticks::EndOfEvent_Simulate

EndOfEvent_Simulate_EGPU_hit
    Opticks GPU num_hit returned from SEvt::GetNumHit_EGPU() immediately following G4CXOpticks::simulate call

EndOfEvent_Simulate_merged_count
EndOfEvent_Simulate_savehit_count
    The sum of merged_count and savehit_count should sum to EGPU_hit in opticksMode:1

EndOfEvent_Simulate_merged_total
EndOfEvent_Simulate_savehit_total
    The totals sum over all events

EndOfEvent_hitCollection_entries0
EndOfEvent_hitCollection_entries1
EndOfEvent_hitCollectionAlt_entries0
EndOfEvent_hitCollectionAlt_entries1
    entries into hitCollection and hitCollectionAlt before and after the
    call to junoSD_PMT_v2_Opticks::EndOfEvent.
    This is recorded from junoSD_PMT_v2::EndOfEvent.

    In all cases entries0 are expected to be zero with entries1
    for the active collection matching the savehit_count

    For opticksMode:1 only hitCollection should be active,
    it contains the GPU originated hits.

    For opticksMode:3 both hitCollection and hitCollectionAlt should
    be active with hitCollection containing CPU hits and
    hitCollectionAlt containing GPU hits.

)LITERAL" ;


    int64_t ProcessHits_count ;
    int64_t ProcessHits_true  ;
    int64_t ProcessHits_false ;
    int64_t SaveNormHit_count ;
    int64_t SaveMuonHit_count ;

    int64_t UNSET ;         // 0
    int64_t NDIS ;          // 1
    int64_t NOPT ;          // 2
    int64_t NEDEP ;         // 3
    int64_t NBOUND ;        // 4
    int64_t NPROC ;         // 5
    int64_t NDETECT ;       // 6
    int64_t NDECULL ;       // 7
    int64_t YMERGE ;        // 8
    int64_t YSAVE ;         // 9
    int64_t SAVENORM ;       // 10
    int64_t SAVEMUON ;       // 11


    int64_t Initialize_G4HCofThisEvent_opticksMode ;
    int64_t Initialize_G4HCofThisEvent_count ;

    int64_t EndOfEvent_Simulate_eventID ;
    int64_t EndOfEvent_Simulate_count ;
    int64_t EndOfEvent_Simulate_EGPU_hit ;
    int64_t EndOfEvent_hitCollection_entries0 ;
    int64_t EndOfEvent_hitCollection_entries1 ;
    int64_t EndOfEvent_hitCollectionAlt_entries0 ;
    int64_t EndOfEvent_hitCollectionAlt_entries1 ;
    int64_t EndOfEvent_Simulate_merged_count ;
    int64_t EndOfEvent_Simulate_savehit_count ;
    int64_t EndOfEvent_Simulate_merged_total ;
    int64_t EndOfEvent_Simulate_savehit_total ;

    SProcessHits_EPH();
    void zero() ;

    int64_t SUM_merged_count_savehit_count() const ;
    int64_t SUM_merged_total_savehit_total() const ;
    std::string desc_kv(const char* key, int64_t value) const ;
    std::string desc() const ;
    void add(int eph, bool processHits);
    void get_kvs( std::vector<std::pair<std::string, int64_t>>& kv ) const ;

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

inline int64_t SProcessHits_EPH::SUM_merged_count_savehit_count() const
{
    return EndOfEvent_Simulate_merged_count + EndOfEvent_Simulate_savehit_count ;
}
inline int64_t SProcessHits_EPH::SUM_merged_total_savehit_total() const
{
    return EndOfEvent_Simulate_merged_total + EndOfEvent_Simulate_savehit_total ;
}

inline std::string SProcessHits_EPH::desc_kv(const char* key, int64_t value) const
{
    int wk = 50 ;
    int wv = 15 ;
    std::stringstream ss ;
    ss << std::setw(wk) << key 
       << std::setw(wv) << value 
       << "       "
       << std::setw(wv) << std::fixed << std::setprecision(6) << double(value)/1.e6   
       << "    /M "
       ;
    std::string str = ss.str();
    return str ;
}

inline std::string SProcessHits_EPH::desc() const
{
    typedef std::pair<std::string, int64_t> SI ;
    typedef std::vector<SI> VSI ;
    VSI kvs ;
    get_kvs(kvs);
    std::stringstream ss ;
    ss << "[SProcessHits_EPH::desc\n" ;
    for(int i=0 ; i < int(kvs.size()) ; i++)
    {
         const char* key = kvs[i].first.c_str();
         int64_t value = kvs[i].second ;

         ss << desc_kv( key, value ) << "\n" ; 

         if(strcmp(key,EndOfEvent_Simulate_savehit_count_)==0)
             ss << desc_kv( SUM_merged_count_savehit_count_, SUM_merged_count_savehit_count()) << "\n" ;

         if(strcmp(key,EndOfEvent_Simulate_savehit_total_)==0)
             ss << desc_kv( SUM_merged_total_savehit_total_, SUM_merged_total_savehit_total()) << "\n" ;
    }
    ss << NOTES ;
    ss << "]SProcessHits_EPH::desc\n" ;
    std::string str = ss.str();
    return str ;
}


inline void SProcessHits_EPH::add(int eph, bool is_hit )
{
    ProcessHits_count  += 1 ;
    ProcessHits_true   += int(is_hit) ;
    ProcessHits_false  += int(!is_hit) ;

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




inline void SProcessHits_EPH::get_kvs( std::vector<std::pair<std::string, int64_t>>& kvs ) const
{
    typedef std::pair<std::string, int64_t> KV ;
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

    std::vector<std::pair<std::string, int64_t>> kvs ;
    get_kvs(kvs);
    meta->set_meta_kv(kvs);
    return meta ;
}


