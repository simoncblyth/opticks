
#include <sstream>
#include "CAction.hh"

const char* CAction::PRE_SAVE_ = "PRE_SAVE" ; 
const char* CAction::POST_SAVE_ = "POST_SAVE" ; 
const char* CAction::PRE_DONE_  = "PRE_DONE" ; 
const char* CAction::POST_DONE_ = "POST_DONE" ; 
const char* CAction::LAST_POST_ = "LAST_POST" ; 
const char* CAction::SURF_ABS_ = "SURF_ABS" ; 
const char* CAction::PRE_SKIP_ = "PRE_SKIP" ; 
const char* CAction::POST_SKIP_ = "POST_SKIP" ; 
const char* CAction::MAT_SWAP_ = "MAT_SWAP" ; 
const char* CAction::STEP_START_ = "STEP_START" ; 
const char* CAction::STEP_REJOIN_ = "STEP_REJOIN" ; 
const char* CAction::STEP_RECOLL_ = "STEP_RECOLL" ; 
const char* CAction::RECORD_TRUNCATE_ = "RECORD_TRUNCATE" ; 
const char* CAction::BOUNCE_TRUNCATE_ = "BOUNCE_TRUNCATE" ; 
const char* CAction::HARD_TRUNCATE_ = "HARD_TRUNCATE" ; 
const char* CAction::ZERO_FLAG_ = "ZERO_FLAG" ; 
const char* CAction::DECREMENT_DENIED_ = "DECREMENT_DENIED" ; 
const char* CAction::TOPSLOT_REWRITE_ = "TOPSLOT_REWRITE" ; 


std::string CAction::Action(int action)
{
    std::stringstream ss ;

    if((action & PRE_SAVE) != 0)  ss << PRE_SAVE_ << " " ; 
    if((action & POST_SAVE) != 0) ss << POST_SAVE_ << " " ; 
    if((action & PRE_DONE) != 0)  ss << PRE_DONE_ << " " ; 
    if((action & POST_DONE) != 0) ss << POST_DONE_ << " " ; 
    if((action & LAST_POST) != 0) ss << LAST_POST_ << " " ; 
    if((action & SURF_ABS) != 0)  ss << SURF_ABS_ << " " ; 
    if((action & PRE_SKIP) != 0)  ss << PRE_SKIP_ << " " ; 
    if((action & POST_SKIP) != 0)  ss << POST_SKIP_ << " " ; 
    if((action & MAT_SWAP) != 0)  ss << MAT_SWAP_ << " " ; 
    if((action & STEP_START) != 0)  ss << STEP_START_ << " " ; 
    if((action & STEP_REJOIN) != 0)  ss << STEP_REJOIN_ << " " ; 
    if((action & STEP_RECOLL) != 0)  ss << STEP_RECOLL_ << " " ; 
    if((action & RECORD_TRUNCATE) != 0)  ss << RECORD_TRUNCATE_ << " " ; 
    if((action & BOUNCE_TRUNCATE) != 0)  ss << BOUNCE_TRUNCATE_ << " " ; 
    if((action & HARD_TRUNCATE) != 0)  ss << HARD_TRUNCATE_ << " " ; 
    if((action & ZERO_FLAG) != 0)  ss << ZERO_FLAG_ << " " ; 
    if((action & DECREMENT_DENIED) != 0)  ss << DECREMENT_DENIED_ << " " ; 
    if((action & TOPSLOT_REWRITE) != 0)  ss << TOPSLOT_REWRITE_ << " " ; 

    return ss.str();
}
 
