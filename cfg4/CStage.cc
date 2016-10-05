#include "CStage.hh"


const char* CStage::UNKNOWN_ = "UNKNOWN" ;
const char* CStage::START_   = "START" ;
const char* CStage::COLLECT_ = "COLLECT" ;
const char* CStage::REJOIN_  = "REJOIN" ;
const char* CStage::RECOLL_  = "RECOLL" ;

const char* CStage::Label( CStage_t stage)
{
    const char* s = 0 ; 
    switch(stage)
    {
        case UNKNOWN:  s = UNKNOWN_ ; break ;
        case START:    s = START_   ; break ;
        case COLLECT:  s = COLLECT_ ; break ;
        case REJOIN:   s = REJOIN_  ; break ;
        case RECOLL:   s = RECOLL_  ; break ;
    } 
    return s ; 
}



