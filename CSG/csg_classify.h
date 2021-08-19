#pragma once

enum { 
     CTRL_RETURN_MISS    = 0,
     CTRL_RETURN_A       = 1,
     CTRL_RETURN_B       = 2,
     CTRL_RETURN_FLIP_B  = 3,
     CTRL_LOOP_A         = 4,   
     CTRL_LOOP_B         = 5
};  


typedef enum { 
     UNDEFINED=0, 
     CONTINUE=1, 
     BREAK=2 
} Action_t ;



#ifdef DEBUG
static const char* CTRL_RETURN_MISS_   = "RETURN_MISS" ; 
static const char* CTRL_RETURN_A_      = "RETURN_A" ; 
static const char* CTRL_RETURN_B_      = "RETURN_B" ; 
static const char* CTRL_RETURN_FLIP_B_ = "RETURN_FLIP_B" ; 
static const char* CTRL_LOOP_A_        = "LOOP_A" ; 
static const char* CTRL_LOOP_B_        = "LOOP_B" ; 
struct CTRL
{
    static const char* Name( int ctrl )
    {
        const char* s = NULL ; 
        switch(ctrl)
        {
            case CTRL_RETURN_MISS:     s = CTRL_RETURN_MISS_   ; break ; 
            case CTRL_RETURN_A:        s = CTRL_RETURN_A_      ; break ; 
            case CTRL_RETURN_B:        s = CTRL_RETURN_B_      ; break ; 
            case CTRL_RETURN_FLIP_B:   s = CTRL_RETURN_FLIP_B_ ; break ; 
            case CTRL_LOOP_A:          s = CTRL_LOOP_A_        ; break ; 
            case CTRL_LOOP_B:          s = CTRL_LOOP_B_        ; break ; 
        }
        return s ; 
    }
};
#endif


typedef enum { 
    State_Enter = 0, 
    State_Exit  = 1, 
    State_Miss  = 2 
} IntersectionState_t ;

#define CSG_CLASSIFY( ise, dir, tmin )   (fabsf((ise).w) > (tmin) ?  ( (ise).x*(dir).x + (ise).y*(dir).y + (ise).z*(dir).z < 0.f ? State_Enter : State_Exit ) : State_Miss )


#ifdef DEBUG
static const char* State_Enter_  = "Enter" ; 
static const char* State_Exit_   = "Exit" ; 
static const char* State_Miss_   = "Miss" ; 
struct IntersectionState
{
    static const char* Name( IntersectionState_t type )
    {
        const char* s = NULL ; 
        switch(type)
        {
            case State_Enter:   s = State_Enter_   ; break ; 
            case State_Exit:    s = State_Exit_    ; break ; 
            case State_Miss:    s = State_Miss_    ; break ; 
        }
        return s ; 
    }
};
#endif


#if defined(__CUDACC__) || defined(__CUDABE__)
   #define LUT_METHOD __device__ __forceinline__ 
#else
   #define LUT_METHOD inline
#endif 


struct LUT
{
    // generated from /Users/blyth/opticks/optixrap/cu by boolean_h.py on Sat Mar  4 20:37:03 2017 
    unsigned packed_boolean_lut_ACloser[4] = { 0x22121141, 0x00014014, 0x00141141, 0x00000000 } ;
    unsigned packed_boolean_lut_BCloser[4] = { 0x22115122, 0x00022055, 0x00133155, 0x00000000 } ;

    LUT_METHOD
    int lookup( OpticksCSG_t operation, IntersectionState_t stateA, IntersectionState_t stateB, bool ACloser )
    {
        const unsigned* lut = ACloser ? packed_boolean_lut_ACloser : packed_boolean_lut_BCloser ;  
        unsigned offset = 3*(unsigned)stateA + (unsigned)stateB ; 
        unsigned index = (unsigned)operation - (unsigned)CSG_UNION ; 
        return offset < 8 ? (( lut[index] >> (offset*4)) & 0xf) : CTRL_RETURN_MISS ;
    }

};


