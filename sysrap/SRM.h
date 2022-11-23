#pragma once
/**
SRM.h : Running Mode
=====================

**/


enum { SRM_UNKNOWN=-1, SRM_DEFAULT, SRM_G4STATE_SAVE, SRM_G4STATE_RERUN } ;  

#include <cstdint>
#include <cstring>

struct SRM
{
    static const char* Name(int32_t mode);
    static int32_t     Type(const char* name);

    static constexpr const char* DEFAULT_ = "SRM_DEFAULT" ;
    static constexpr const char* G4STATE_SAVE_ = "SRM_G4STATE_SAVE" ;
    static constexpr const char* G4STATE_RERUN_ = "SRM_G4STATE_RERUN" ;
};
inline const char* SRM::Name(int32_t mode)
{
    const char* s = nullptr ;
    switch(mode)
    {
        case SRM_DEFAULT:       s = DEFAULT_       ; break ;
        case SRM_G4STATE_SAVE:  s = G4STATE_SAVE_  ; break ;
        case SRM_G4STATE_RERUN: s = G4STATE_RERUN_ ; break ;
    }
    return s ; 
}
inline int32_t SRM::Type(const char* name)
{
    int32_t type = SRM_UNKNOWN ;
    if(strcmp(name,DEFAULT_) == 0 )       type = SRM_DEFAULT      ;
    if(strcmp(name,G4STATE_SAVE_) == 0 )  type = SRM_G4STATE_SAVE ;
    if(strcmp(name,G4STATE_RERUN_) == 0 ) type = SRM_G4STATE_RERUN ;

    assert( type != SRM_UNKNOWN ); 
    return type ;
}

