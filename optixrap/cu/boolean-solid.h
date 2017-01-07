#pragma once

// see csg-;csg-vi for notes

enum 
{
    ReturnMiss       = 0x1 << 0,
    ReturnAIfCloser  = 0x1 << 1,
    ReturnAIfFarther = 0x1 << 2,
    ReturnA          = 0x1 << 3,
    ReturnBIfCloser  = 0x1 << 4,
    ReturnBIfFarther = 0x1 << 5,
    ReturnB          = 0x1 << 6,
    FlipB            = 0x1 << 7,
    AdvanceAAndLoop  = 0x1 << 8,
    AdvanceBAndLoop  = 0x1 << 9,
    AdvanceAAndLoopIfCloser = 0x1 << 10,    
    AdvanceBAndLoopIfCloser = 0x1 << 11
};


#ifdef BOOLEAN_SOLID_DEBUG
#include <string>
#include <sstream>
#include <iomanip>

static const char* ReturnMiss_ = "ReturnMiss" ;
static const char* ReturnAIfCloser_ = "ReturnAIfCloser" ;
static const char* ReturnAIfFarther_ = "ReturnAIfFarther" ;
static const char* ReturnA_ = "ReturnA" ;

static const char* ReturnBIfCloser_ = "ReturnBIfCloser" ;
static const char* ReturnBIfFarther_ = "ReturnBIfFarther" ;
static const char* ReturnB_ = "ReturnB" ;

static const char* FlipB_ = "FlipB" ;
static const char* AdvanceAAndLoop_ = "AdvanceAAndLoop" ;
static const char* AdvanceBAndLoop_ = "AdvanceBAndLoop" ;
static const char* AdvanceAAndLoopIfCloser_ = "AdvanceAAndLoopIfCloser" ;
static const char* AdvanceBAndLoopIfCloser_ = "AdvanceBAndLoopIfCloser" ;

static const char* Enter_ = "Enter" ; 
static const char* Exit_ = "Exit" ; 
static const char* Miss_ = "Miss" ; 

#endif


struct Union ;
struct Difference ; 
struct Intersection ; 

typedef enum { Enter, Exit, Miss } IntersectionState_t ;


#ifndef __CUDACC__

template <class T>
struct boolean_action
{
   static int _table[9] ; 
   int operator()( IntersectionState_t a, IntersectionState_t b );

#ifdef BOOLEAN_SOLID_DEBUG
   std::string dumptable( const char* msg );
#endif

};

#endif



#ifdef BOOLEAN_SOLID_DEBUG

const char* description( IntersectionState_t x )
{
   const char* s = NULL ; 
   switch(x)
   {
      case Enter: s = Enter_ ; break ; 
      case Exit:  s = Exit_ ; break ; 
      case Miss:  s = Miss_ ; break ; 
   }
   return s ; 
}

std::string description( int action )
{
    std::stringstream ss ; 

    if(action & ReturnMiss ) ss << ReturnMiss_ << " " ; 

    if(action & ReturnAIfCloser ) ss << ReturnAIfCloser_ << " " ; 
    if(action & ReturnAIfFarther ) ss << ReturnAIfFarther_ << " " ; 
    if(action & ReturnA ) ss << ReturnA_ << " " ; 

    if(action & ReturnBIfCloser ) ss << ReturnBIfCloser_ << " " ; 
    if(action & ReturnBIfFarther ) ss << ReturnBIfFarther_ << " " ; 
    if(action & ReturnB ) ss << ReturnB_ << " " ; 

    if(action & FlipB ) ss << FlipB_ << " " ; 
    if(action & AdvanceAAndLoop ) ss << AdvanceAAndLoop_ << " " ; 
    if(action & AdvanceBAndLoop ) ss << AdvanceBAndLoop_ << " " ; 
    if(action & AdvanceAAndLoopIfCloser ) ss << AdvanceAAndLoopIfCloser_ << " " ; 
    if(action & AdvanceBAndLoopIfCloser ) ss << AdvanceBAndLoopIfCloser_ << " " ; 

    return ss.str();    
}



template<class T>
std::string boolean_action<T>::dumptable(const char* msg)
{
    std::stringstream ss ; 
    ss << msg << std::endl ; 

    for(int ia=0 ; ia < 3 ; ia++)
    {
        IntersectionState_t a = (IntersectionState_t)ia ; 
        for(int ib=0 ; ib < 3 ; ib++)
        {
            IntersectionState_t b = (IntersectionState_t)ib ; 
   
            int action = (*this)(a, b) ; 

            std::string action_desc = description(action); 

            ss
                << std::setw(5) << description(a) << "A "
                << std::setw(5) << description(b) << "B "
                << " -> " 
                <<  action_desc.c_str()
                << std::endl ; 
        }
    }
    return ss.str(); 
}

#endif


enum 
{
    Union_EnterA_EnterB = ReturnAIfCloser | ReturnBIfCloser,
    Union_EnterA_ExitB  = ReturnBIfCloser | AdvanceAAndLoop,
    Union_EnterA_MissB  = ReturnA, 
    Union_ExitA_EnterB  = ReturnAIfCloser | AdvanceBAndLoop,
    Union_ExitA_ExitB   = ReturnAIfFarther | ReturnBIfFarther,
    Union_ExitA_MissB   = ReturnA ,
    Union_MissA_EnterB  = ReturnB ,
    Union_MissA_ExitB   = ReturnB ,
    Union_MissA_MissB   = ReturnMiss 
};

enum 
{
    Difference_EnterA_EnterB =  ReturnAIfCloser | AdvanceBAndLoop,
    Difference_EnterA_ExitB  =  AdvanceAAndLoopIfCloser | AdvanceBAndLoopIfCloser,
    Difference_EnterA_MissB  =  ReturnA,
    Difference_ExitA_EnterB  =  ReturnAIfCloser | ReturnBIfCloser | FlipB,
    Difference_ExitA_ExitB   =  ReturnBIfCloser | FlipB | AdvanceAAndLoop,
    Difference_ExitA_MissB   =  ReturnA,
    Difference_MissA_EnterB  =  ReturnMiss,
    Difference_MissA_ExitB   =  ReturnMiss,
    Difference_MissA_MissB   =  ReturnMiss
};

enum 
{
    Intersection_EnterA_EnterB = AdvanceAAndLoopIfCloser | AdvanceBAndLoopIfCloser,
    Intersection_EnterA_ExitB  = ReturnAIfCloser | AdvanceBAndLoop,
    Intersection_EnterA_MissB  = ReturnMiss,
    Intersection_ExitA_EnterB  = ReturnBIfCloser | AdvanceAAndLoop,
    Intersection_ExitA_ExitB   = ReturnAIfCloser | ReturnBIfCloser,
    Intersection_ExitA_MissB   = ReturnMiss,
    Intersection_MissA_EnterB  = ReturnMiss, 
    Intersection_MissA_ExitB   = ReturnMiss,
    Intersection_MissA_MissB   = ReturnMiss 
};



#ifndef __CUDACC__

template<>
int boolean_action<Union>::_table[9] = 
    { 
         Union_EnterA_EnterB, Union_EnterA_ExitB, Union_EnterA_MissB, 
         Union_ExitA_EnterB, Union_ExitA_ExitB, Union_ExitA_MissB,
         Union_MissA_EnterB, Union_MissA_ExitB, Union_MissA_MissB 
    } ;

template<>
int boolean_action<Difference>::_table[9] = 
     { 
         Difference_EnterA_EnterB, Difference_EnterA_ExitB, Difference_EnterA_MissB, 
         Difference_ExitA_EnterB, Difference_ExitA_ExitB, Difference_ExitA_MissB,
         Difference_MissA_EnterB, Difference_MissA_ExitB, Difference_MissA_MissB 
     } ;

template<>
int boolean_action<Intersection>::_table[9] = 
      { 
          Intersection_EnterA_EnterB, Intersection_EnterA_ExitB, Intersection_EnterA_MissB, 
          Intersection_ExitA_EnterB, Intersection_ExitA_ExitB, Intersection_ExitA_MissB,
          Intersection_MissA_EnterB, Intersection_MissA_ExitB, Intersection_MissA_MissB 
      } ;


template<class T>
int boolean_action<T>::operator()( IntersectionState_t stateA, IntersectionState_t stateB )
{
    int a = (int)stateA ; 
    int b = (int)stateB ; 
    int offset = 3*a + b ;   
    return _table[offset] ; 
}

#endif



__host__
__device__
int union_action( IntersectionState_t stateA, IntersectionState_t stateB  )
{
    int offset = 3*(int)stateA + (int)stateB ;   
    int action = Union_MissA_MissB ; 
    switch(offset)
    {
       case 0: action=Union_EnterA_EnterB ; break ; 
       case 1: action=Union_EnterA_ExitB  ; break ; 
       case 2: action=Union_EnterA_MissB  ; break ; 
       case 3: action=Union_ExitA_EnterB  ; break ; 
       case 4: action=Union_ExitA_ExitB   ; break ; 
       case 5: action=Union_ExitA_MissB   ; break ; 
       case 6: action=Union_MissA_EnterB  ; break ; 
       case 7: action=Union_MissA_ExitB   ; break ; 
       case 8: action=Union_MissA_MissB   ; break ; 
    }
    return action ; 
}

__host__
__device__
int intersection_action( IntersectionState_t stateA, IntersectionState_t stateB  )
{
    int offset = 3*(int)stateA + (int)stateB ;   
    int action = Intersection_MissA_MissB ; 
    switch(offset)
    {
       case 0: action=Intersection_EnterA_EnterB ; break ; 
       case 1: action=Intersection_EnterA_ExitB  ; break ; 
       case 2: action=Intersection_EnterA_MissB  ; break ; 
       case 3: action=Intersection_ExitA_EnterB  ; break ; 
       case 4: action=Intersection_ExitA_ExitB   ; break ; 
       case 5: action=Intersection_ExitA_MissB   ; break ; 
       case 6: action=Intersection_MissA_EnterB  ; break ; 
       case 7: action=Intersection_MissA_ExitB   ; break ; 
       case 8: action=Intersection_MissA_MissB   ; break ; 
    }
    return action ; 
}


__host__
__device__
int difference_action( IntersectionState_t stateA, IntersectionState_t stateB  )
{
    int offset = 3*(int)stateA + (int)stateB ;   
    int action = Difference_MissA_MissB ; 
    switch(offset)
    {
       case 0: action=Difference_EnterA_EnterB ; break ; 
       case 1: action=Difference_EnterA_ExitB  ; break ; 
       case 2: action=Difference_EnterA_MissB  ; break ; 
       case 3: action=Difference_ExitA_EnterB  ; break ; 
       case 4: action=Difference_ExitA_ExitB   ; break ; 
       case 5: action=Difference_ExitA_MissB   ; break ; 
       case 6: action=Difference_MissA_EnterB  ; break ; 
       case 7: action=Difference_MissA_ExitB   ; break ; 
       case 8: action=Difference_MissA_MissB   ; break ; 
    }
    return action ; 
}


