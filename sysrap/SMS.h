#pragma once

/**
SMS.h : Saving Mode
=====================

Configure absolute or relative directory saving of SEvt NumPy arrays and metadata.
Absolute normally under $TMP and relative under the invoking directory::

    export OPTICKS_MODE_SAVE=SMS_ABSOLUTE  ## default
    export OPTICKS_MODE_SAVE=SMS_RELATIVE

Configure what arrays are saved::

    export OPTICKS_EVENT_MODE=Minimal  # default, nothing saved
    export OPTICKS_EVENT_MODE=Hit      # Hit and Genstep
    export OPTICKS_EVENT_MODE=DebugLite

    ## see SEventConfig::Initialize_Comp_Simulate_ for details

**/


enum { SMS_UNKNOWN=-1,
       SMS_ABSOLUTE,
       SMS_RELATIVE
       } ;

#include <cstdint>
#include <cstring>

struct SMS
{
    static const char* Name(int32_t mode);
    static int32_t     Type(const char* name);

    static constexpr const char* SMS_UNKNOWN_  = "SMS_UNKNOWN" ;
    static constexpr const char* SMS_ABSOLUTE_ = "SMS_ABSOLUTE" ;
    static constexpr const char* SMS_RELATIVE_ = "SMS_RELATIVE" ;
};
inline const char* SMS::Name(int32_t mode)
{
    const char* s = nullptr ;
    switch(mode)
    {
        case SMS_UNKNOWN:   s = SMS_UNKNOWN_   ; break ;
        case SMS_ABSOLUTE:  s = SMS_ABSOLUTE_  ; break ;
        case SMS_RELATIVE:  s = SMS_RELATIVE_  ; break ;
    }
    return s ;
}
inline int32_t SMS::Type(const char* name)
{
    int32_t type = SMS_UNKNOWN ;
    if(strcmp(name,SMS_UNKNOWN_) == 0 )  type = SMS_UNKNOWN  ;
    if(strcmp(name,SMS_ABSOLUTE_) == 0 ) type = SMS_ABSOLUTE ;
    if(strcmp(name,SMS_RELATIVE_) == 0 ) type = SMS_RELATIVE ;
    assert( type != SMS_UNKNOWN );
    return type ;
}

