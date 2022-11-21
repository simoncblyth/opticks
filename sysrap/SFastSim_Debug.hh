#pragma once
/**
SFastSim_Debug.h
===========================


**/

#include "plog/Severity.h"
#include <vector>
#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SFastSim_Debug
{   
    static const plog::Severity LEVEL ; 
    static std::vector<SFastSim_Debug> record ;   
    static constexpr const unsigned NUM_QUAD = 4u ; 
    static constexpr const char* NAME = "SFastSim_Debug.npy" ; 
    static constexpr int LIMIT = 100000 ; 
    static void Save(const char* dir); 
    void add(); 
    void fill(double value); 

    double posx ;  // pos = fs[:,0,:3]
    double posy ;
    double posz ;
    double time ;  // tim = fs[:,0,3]

    double dirx ;  // mom = fs[:,1,:3]
    double diry ;
    double dirz ;
    double dist1 ; // ds1 = fs[:,1,3]

    double polx ;  // pol = fs[:,2,:3]
    double poly ;
    double polz ;
    double dist2 ; // ds2 = fs[:,2,3]

    double ModelTrigger ; // trg = fs[:,3,0].astype(np.int64)   ## wasting 63 bits 
    double whereAmI ;     // wai = fs[:,3,1].astype(np.int64)   ## wasting 62-63 bits 
    double c ;            // c   = fs[:,3,2]
    double PhotonId ;     // pid = fs[:,3,3].astype(np.int64)   ## again wasting bits 

    // NB for python parsing check line terminations with set list
    // TODO: improve py parsing to cope with comments
};

