#pragma once
/**
sProcessHits_EPH.h
====================

**/


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
       NDECOLL,      // 10
       SAVENORM,     // 11
       SAVEMUON      // 12
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
    static constexpr const char* NDECOLL_ = "EPH_NDECOLL" ; // 10 
    static constexpr const char* SAVENORM_ = "EPH_SAVENORM" ; // 11
    static constexpr const char* SAVEMUON_ = "EPH_SAVEMUON" ; // 12

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
            case NDECOLL: sn = NDECOLL_  ; break ;  // 10
            case SAVENORM: sn = SAVENORM_  ; break ;  // 11
            case SAVEMUON: sn = SAVEMUON_  ; break ;  // 12
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
        names.push_back(Name(NDECOLL)); // 10
        names.push_back(Name(SAVENORM)); // 11
        names.push_back(Name(SAVEMUON)); // 12
    }
    // end of EPH namespace
}


