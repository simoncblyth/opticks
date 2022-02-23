#pragma once

/**
CSGName.h
===========

Identity machinery using the foundry list of meshnames (aka solid names) 

**/

#include <vector>
#include <string>
#include "plog/Severity.h"

#include "CSG_API_EXPORT.hh"

struct CSG_API CSGName
{
    static const plog::Severity LEVEL ; 
    const CSGFoundry* foundry ; 
    const std::vector<std::string>& name ; 
    static int ParseIntString(const char* arg, int fallback=-1);
    static void ParseSOPR(int& solidIdx, int& primIdxRel, const char* sopr ); 


    CSGName( const CSGFoundry* foundry );  

    unsigned getNumName() const;
    const char* getName(unsigned idx) const ;
    const char* getAbbr(unsigned idx) const ;

    int getIndex( const char* name    , unsigned& count) const ;
    int findIndex(const char* starting, unsigned& count, int max_count=-1) const ;

    static const char* parseArg_ALL ; 
    int parseArg(const char* arg, unsigned& count ) const ;
    void parseMOI(int& midx, int& mord, int& iidx, const char* moi) const ;


}; 
