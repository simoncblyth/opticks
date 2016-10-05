#pragma once

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API CStage 
{
    public:
    typedef enum { UNKNOWN, START, COLLECT, REJOIN, RECOLL } CStage_t ;
    static const char* UNKNOWN_ ;
    static const char* START_  ;
    static const char* COLLECT_  ;
    static const char* REJOIN_  ;
    static const char* RECOLL_  ;
    static const char* Label( CStage_t stage);
};

#include "CFG4_TAIL.hh"

