#pragma once
/**
SRM.h : Running Mode
=====================

HMM: if needed to combine these modes would need to split or 
change this to using bitmasks 

**/


enum { SRM_UNKNOWN=-1, SRM_DEFAULT, SRM_G4STATE_SAVE, SRM_G4STATE_RERUN, SRM_TORCH, SRM_INPHO, SRM_GUN } ;  

#include <cstdint>
#include <cstring>

struct SRM
{
    static const char* Name(int32_t mode);
    static int32_t     Type(const char* name);

    static constexpr const char* DEFAULT_ = "SRM_DEFAULT" ;
    static constexpr const char* G4STATE_SAVE_ = "SRM_G4STATE_SAVE" ;
    static constexpr const char* G4STATE_RERUN_ = "SRM_G4STATE_RERUN" ;
    static constexpr const char* TORCH_ = "SRM_TORCH" ;
    static constexpr const char* INPHO_ = "SRM_INPHO" ;
    static constexpr const char* GUN_ = "SRM_GUN" ;
};
inline const char* SRM::Name(int32_t mode)
{
    const char* s = nullptr ;
    switch(mode)
    {
        case SRM_DEFAULT:       s = DEFAULT_       ; break ;
        case SRM_G4STATE_SAVE:  s = G4STATE_SAVE_  ; break ;
        case SRM_G4STATE_RERUN: s = G4STATE_RERUN_ ; break ;
        case SRM_TORCH        : s = TORCH_         ; break ;
        case SRM_INPHO        : s = INPHO_         ; break ;
        case SRM_GUN          : s = GUN_           ; break ;
    }
    return s ; 
}
inline int32_t SRM::Type(const char* name)
{
    int32_t type = SRM_UNKNOWN ;
    if(strcmp(name,DEFAULT_) == 0 )       type = SRM_DEFAULT      ;
    if(strcmp(name,G4STATE_SAVE_) == 0 )  type = SRM_G4STATE_SAVE ;
    if(strcmp(name,G4STATE_RERUN_) == 0 ) type = SRM_G4STATE_RERUN ;
    if(strcmp(name,TORCH_) == 0 )         type = SRM_TORCH ;
    if(strcmp(name,INPHO_) == 0 )         type = SRM_INPHO ;
    if(strcmp(name,GUN_) == 0 )           type = SRM_GUN ;

    assert( type != SRM_UNKNOWN ); 
    return type ;
}

