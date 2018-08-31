#pragma once

#include "NPY_API_EXPORT.hh"
#include "OpticksCSG.h"
#include <string>

// minimal node class standin for nnode used to develop tree machinery 

struct NPY_API no
{
    static no* copy( const no* a ); // retaining vtable of subclass instances 
    no* make_copy() const ;         // retaining vtable of subclass instances

    const char* label ; 
    no* left ; 
    no* right ; 
    no* parent ; 

    unsigned depth ;
    unsigned subdepth ;
    OpticksCSG_t type ; 
    bool    complement ; 
 
    std::string id() const ;
    std::string desc() const ;
    unsigned maxdepth() const ;
    unsigned _maxdepth(unsigned depth) const;   // recursive 
    static unsigned nmaxu(const unsigned a, const unsigned b);

    bool is_primitive() const ; 
    bool is_bileaf() const ; 
    bool is_operator() const ; 
    bool is_zero() const ; 

    bool is_lrzero() const ;  //  l-zero AND  r-zero
    bool is_rzero() const ;   // !l-zero AND  r-zero
    bool is_lzero() const ;   //  l-zero AND !r-zero

    static no make_node( OpticksCSG_t type, no* left=NULL, no* right=NULL );  
    static no* make_operator_ptr(OpticksCSG_t operator_, no* left=NULL, no* right=NULL );

}; 


inline no* no::make_operator_ptr(OpticksCSG_t operator_, no* left, no* right )
{
    no* node = new no(make_node(operator_, left , right ));
    return node ; 
}




