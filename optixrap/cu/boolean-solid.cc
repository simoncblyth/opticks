//
// boolean-solid.cc was just used to check boolean-solid.h 
// in which the boolean action tables are implemented
//
// initially tried to use template specialization
// to lay down the action tables into class members
// with nvcc, but gave up and adopted the trivial
// case statement approach 
// 
//
// boolean-solid.cc is not integrated with anything, 
// build and run with::
//

/*
clang boolean-solid.cc -lstdc++ -I$OPTICKS_HOME/optickscore && ./a.out && rm a.out
*/


#include <cstddef>
#include "OpticksShape.h"
#include "boolean-solid.h"

#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>


// these structs have no implementation, just names
struct Union ;
struct Difference ; 
struct Intersection ; 


static const char* ReturnMiss_ = "ReturnMiss" ;
static const char* ReturnAIfCloser_ = "ReturnAIfCloser" ;
static const char* ReturnAIfFarther_ = "ReturnAIfFarther" ;
static const char* ReturnA_ = "ReturnA" ;

static const char* ReturnBIfCloser_ = "ReturnBIfCloser" ;
static const char* ReturnFlipBIfCloser_ = "ReturnFlipBIfCloser" ;
static const char* ReturnBIfFarther_ = "ReturnBIfFarther" ;
static const char* ReturnB_ = "ReturnB" ;

static const char* AdvanceAAndLoop_ = "AdvanceAAndLoop" ;
static const char* AdvanceBAndLoop_ = "AdvanceBAndLoop" ;
static const char* AdvanceAAndLoopIfCloser_ = "AdvanceAAndLoopIfCloser" ;
static const char* AdvanceBAndLoopIfCloser_ = "AdvanceBAndLoopIfCloser" ;

static const char* Enter_ = "Enter" ; 
static const char* Exit_ = "Exit" ; 
static const char* Miss_ = "Miss" ; 


template <class T>
struct boolean_action
{
   static int _table[9] ; 
   int operator()( IntersectionState_t a, IntersectionState_t b );
   std::string dumptable( const char* msg );
};


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
    if(action & ReturnFlipBIfCloser ) ss << ReturnFlipBIfCloser_ << " " ; 
    if(action & ReturnBIfFarther ) ss << ReturnBIfFarther_ << " " ; 
    if(action & ReturnB ) ss << ReturnB_ << " " ; 

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


std::string dump_lookup(const char* msg)
{
    std::stringstream ss ; 
    ss << msg << std::endl ; 
    for(int io=0 ; io < 3 ; io++)
    {
       OpticksCSG_t o = (OpticksCSG_t)io ; 
       ss << CSGName(o) << std::endl ; 
       for(int ia=0 ; ia < 3 ; ia++)
       {
           for(int ib=0 ; ib < 3 ; ib++)
           {
               IntersectionState_t a = (IntersectionState_t)ia ; 
               IntersectionState_t b = (IntersectionState_t)ib ;
               int action = boolean_lookup(o, a, b );
               std::string action_desc = description(action); 
               ss
                   << std::setw(5) << description(a) << "A "
                   << std::setw(5) << description(b) << "B "
                   << " -> " 
                   <<  action_desc.c_str()
                   << std::endl ; 
            }
       }
    }
    return ss.str(); 
}


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







int main(int argc, char** argv)
{
    boolean_action<Union>        uat ; 
    boolean_action<Intersection> iat ; 
    boolean_action<Difference>   dat ; 

    std::cout << uat.dumptable("Union") << std::endl ;
    std::cout << iat.dumptable("Intersection") << std::endl ;
    std::cout << dat.dumptable("Difference") << std::endl ;

    std::cout << dump_lookup("dump_lookup") << std::endl ; 

    return 0 ; 
}
