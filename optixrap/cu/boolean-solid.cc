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


static const char* CTRL_RETURN_MISS_   = "CTRL_RETURN_MISS" ; 
static const char* CTRL_RETURN_A_      = "CTRL_RETURN_A" ;
static const char* CTRL_RETURN_B_      = "CTRL_RETURN_B" ;
static const char* CTRL_RETURN_FLIP_B_ = "CTRL_RETURN_FLIP_B" ;
static const char* CTRL_LOOP_A_        = "CTRL_LOOP_A" ;
static const char* CTRL_LOOP_B_        = "CTRL_LOOP_B" ;


static const char* ERROR_LHS_POP_EMPTY_ = "ERROR_LHS_POP_EMPTY" ;
static const char* ERROR_RHS_POP_EMPTY_ = "ERROR_RHS_POP_EMPTY" ;
static const char* ERROR_LHS_END_NONEMPTY_ = "ERROR_LHS_END_NONEMPTY" ;
static const char* ERROR_RHS_END_EMPTY_ = "ERROR_RHS_END_EMPTY" ;
static const char* ERROR_BAD_CTRL_ = "ERROR_BAD_CTRL" ;
static const char* ERROR_LHS_OVERFLOW_ = "ERROR_LHS_OVERFLOW" ;
static const char* ERROR_RHS_OVERFLOW_ = "ERROR_RHS_OVERFLOW" ;
static const char* ERROR_LHS_TRANCHE_OVERFLOW_ = "ERROR_LHS_TRANCHE_OVERFLOW" ;
static const char* ERROR_RHS_TRANCHE_OVERFLOW_ = "ERROR_RHS_TRANCHE_OVERFLOW" ;







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


std::string desc_ctrl( int ctrl )
{
    std::stringstream ss ; 
    if(ctrl & CTRL_RETURN_MISS ) ss << CTRL_RETURN_MISS_ << " " ; 
    if(ctrl & CTRL_RETURN_A ) ss << CTRL_RETURN_A_ << " " ; 
    if(ctrl & CTRL_RETURN_B ) ss << CTRL_RETURN_B_ << " " ; 
    if(ctrl & CTRL_RETURN_FLIP_B ) ss << CTRL_RETURN_FLIP_B_ << " " ; 

    if(ctrl & CTRL_LOOP_A ) ss << CTRL_LOOP_A_ << " " ; 
    if(ctrl & CTRL_LOOP_B ) ss << CTRL_LOOP_B_ << " " ; 
    return ss.str();    
}

std::string desc_err( long err )
{
    std::stringstream ss ; 
    if(err & ERROR_LHS_POP_EMPTY ) ss << ERROR_LHS_POP_EMPTY_ << " " ;
    if(err & ERROR_RHS_POP_EMPTY ) ss << ERROR_RHS_POP_EMPTY_ << " " ;
    if(err & ERROR_LHS_END_NONEMPTY ) ss << ERROR_LHS_END_NONEMPTY_ << " " ;
    if(err & ERROR_RHS_END_EMPTY ) ss << ERROR_RHS_END_EMPTY_ << " " ;
    if(err & ERROR_BAD_CTRL ) ss << ERROR_BAD_CTRL_ << " " ;
    if(err & ERROR_LHS_OVERFLOW ) ss << ERROR_LHS_OVERFLOW_ << " " ;
    if(err & ERROR_RHS_OVERFLOW ) ss << ERROR_RHS_OVERFLOW_ << " " ;
    if(err & ERROR_LHS_TRANCHE_OVERFLOW ) ss << ERROR_LHS_TRANCHE_OVERFLOW_ << " " ;
    if(err & ERROR_RHS_TRANCHE_OVERFLOW ) ss << ERROR_RHS_TRANCHE_OVERFLOW_ << " " ;
    return ss.str();    
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




std::string dump_ctrl_enum(const char* msg="dump_ctrl_enum")
{
    std::stringstream ss ; 
    ss << msg << std::endl ; 

    for(unsigned i=0 ; i < 16 ; i++)
    {   
        unsigned ctrl = 0x1 << i ; 
        ss 
           << std::setw(5) << i
           << std::setw(5) << std::hex << ctrl << std::dec
           << std::setw(30) <<  desc_ctrl(ctrl)
           << std::endl ;  
    }
    return ss.str(); 
}



std::string dump_action_enum(const char* msg="dump_action_enum")
{
    std::stringstream ss ; 
    ss << msg << std::endl ; 

    for(unsigned i=0 ; i < 16 ; i++)
    {   
        unsigned action = 0x1 << i ; 
        ss 
           << std::setw(5) << i
           << std::setw(5) << std::hex << action << std::dec
           << std::setw(30) <<  description(action)
           << std::endl ;  
    }
    return ss.str(); 
}

std::string dump_error_enum(const char* msg="dump_error_enum")
{
    std::stringstream ss ; 
    ss << msg << std::endl ; 
    for(unsigned i=0 ; i < 10 ; i++)
    {
        ss << std::setw(5) << i << " " <<  desc_err(i) << std::endl ; 
    }
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


/*
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
                   << " 0x" << std::hex << std::setw(5) << action << std::dec 
                   << " "
                   <<  action_desc.c_str()
                   << std::endl ; 
            }
       }
    }
    return ss.str(); 
}
*/




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

    std::cout << dump_action_enum() << std::endl ; 
    std::cout << dump_ctrl_enum() << std::endl ; 
    std::cout << dump_error_enum() << std::endl ; 

    std::cout << uat.dumptable("Union") << std::endl ;
    std::cout << iat.dumptable("Intersection") << std::endl ;
    std::cout << dat.dumptable("Difference") << std::endl ;

    //std::cout << dump_lookup("dump_lookup") << std::endl ; 

    for(int i=1 ; i < argc ; i++)
    {
        long code = strtol( argv[i], NULL, 0);
        std::cout << std::setw(10) << argv[i] << " -> " << std::hex << code << std::dec << " -> " << desc_err(code) << std::endl ;           
    }


    return 0 ; 
}
