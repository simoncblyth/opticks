#pragma once
/**
SGLFW_Keys.h : record of keyboard keys currently held down with modifiers bitfield summarization
================================================================================================== 

**/

#include "SGLM_Modifiers.h"

struct SGLFW_Keys
{
    enum { NUM_KEYS = 512 } ;
    bool down[NUM_KEYS] ; 

    SGLFW_Keys();  
    void key_pressed(unsigned key);
    void key_released(unsigned key);
    unsigned modifiers() const ; 
};

inline SGLFW_Keys::SGLFW_Keys()
{
   for(int i=0 ; i < NUM_KEYS ; i++) down[i] = false ; 
}
inline void SGLFW_Keys::key_pressed(unsigned key)
{
    if(key < NUM_KEYS) down[key] = true ; 
}
inline void SGLFW_Keys::key_released(unsigned key)
{ 
    if(key < NUM_KEYS) down[key] = false ; 
}
/**
SGLFW_Keys::modifiers
----------------------

Returns bitfield representing a restricted selection of "controller" keys 
that are currently held down.
 
NB these keys should be kept distinct from toggle keys.  

**/
inline unsigned SGLFW_Keys::modifiers() const 
{
    unsigned modifiers = 0 ;
    if( down[GLFW_KEY_LEFT_SHIFT]   || down[GLFW_KEY_RIGHT_SHIFT] )    modifiers += SGLM_Modifiers::MOD_SHIFT ; 
    if( down[GLFW_KEY_LEFT_CONTROL] || down[GLFW_KEY_RIGHT_CONTROL] )  modifiers += SGLM_Modifiers::MOD_CONTROL ; 
    if( down[GLFW_KEY_LEFT_ALT]     || down[GLFW_KEY_RIGHT_ALT] )      modifiers += SGLM_Modifiers::MOD_ALT ; 
    if( down[GLFW_KEY_LEFT_SUPER]   || down[GLFW_KEY_RIGHT_SUPER] )    modifiers += SGLM_Modifiers::MOD_SUPER ; 

    if( down[GLFW_KEY_W] ) modifiers += SGLM_Modifiers::MOD_W ; 
    if( down[GLFW_KEY_A] ) modifiers += SGLM_Modifiers::MOD_A ; 
    if( down[GLFW_KEY_S] ) modifiers += SGLM_Modifiers::MOD_S ; 
    if( down[GLFW_KEY_D] ) modifiers += SGLM_Modifiers::MOD_D ; 
    if( down[GLFW_KEY_Q] ) modifiers += SGLM_Modifiers::MOD_Q ; 
    if( down[GLFW_KEY_E] ) modifiers += SGLM_Modifiers::MOD_E ; 
    if( down[GLFW_KEY_R] ) modifiers += SGLM_Modifiers::MOD_R ; 
    if( down[GLFW_KEY_Y] ) modifiers += SGLM_Modifiers::MOD_Y ; 

    return modifiers ; 
}


