#pragma once

#include <vector>
#include <string>


struct CSGName
{
    const CSGFoundry* foundry ; 
    const std::vector<std::string>& name ; 
    static int ParseIntString(const char* arg, int fallback=-1);

    CSGName( const CSGFoundry* foundry );  

    unsigned getNumName() const;
    const char* getName(unsigned idx) const ;
    const char* getAbbr(unsigned idx) const ;

    int getIndex( const char* name    , unsigned& count) const ;
    int findIndex(const char* starting, unsigned& count, unsigned max_count=-1) const ;

    int parseArg(const char* arg, unsigned& count ) const ;
    void parseMOI(int& midx, int& mord, int& iidx, const char* moi) const ;


}; 
