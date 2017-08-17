#pragma once

/**
OGeoStat
=====

**/

#include <string>
#include "OXRAP_API_EXPORT.hh"

struct OXRAP_API OGeoStat 
{
    unsigned      mmIndex ; 
    unsigned      numPrim ;
    unsigned      numPart ;
    unsigned      numTran ;
    unsigned      numPlan ;

    OGeoStat( unsigned mmIndex_, unsigned numPrim_, unsigned numPart_, unsigned numTran_, unsigned numPlan_ );
    std::string desc();

};




