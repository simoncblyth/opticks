#pragma once

struct SDBG
{
    enum { NONE, BACKTRACE, SUMMARY, CALLER, INTERRUPT } ; 
    
    static constexpr const char* BACKTRACE_ = "backtrace" ; 
    static constexpr const char* SUMMARY_ = "summary" ; 
    static constexpr const char* CALLER_ = "caller" ; 
    static constexpr const char* INTERRUPT_ = "interrupt" ; 

    static const char* Name(unsigned action); 
    static unsigned    Action(const char* ); 

}; 

inline const char* SDBG::Name(unsigned action)
{
    const char* s = nullptr ; 
    switch(action)
    {
        case BACKTRACE: s = BACKTRACE_ ; break ; 
        case SUMMARY:   s = SUMMARY_   ; break ; 
        case CALLER:    s = CALLER_    ; break ; 
        case INTERRUPT: s = INTERRUPT_ ; break ; 
    }
    return s ; 
}

inline unsigned SDBG::Action(const char* action_)
{
    unsigned action = NONE ; 
    if(strcmp(action_, BACKTRACE_) == 0 ) action = BACKTRACE ; 
    if(strcmp(action_, SUMMARY_) == 0 )   action = SUMMARY ; 
    if(strcmp(action_, CALLER_) == 0 )    action = CALLER ; 
    if(strcmp(action_, INTERRUPT_) == 0 ) action = INTERRUPT ; 
    return action ; 
}


