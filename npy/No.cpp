
#include <sstream>
#include <iomanip>
#include "No.hpp"


no* no::copy( const no* a )  // static  ... matches nnode::copy where this is actually needed
{
    no* c = NULL ; 
    c = new no(*a) ; 
    return c ; 
}
no* no::make_copy() const 
{
    return no::copy(this); 
}


std::string no::id() const 
{
    std::stringstream ss ; 
    ss  
        << ( complement ? "!" : "" )
        << ( label ? label : "" )
        ;     
    return ss.str();
}
 

std::string no::desc() const 
{
    int w = 2 ; 
    std::stringstream ss ; 
    ss 
       << std::setw(w) << label 
       << " l " << std::setw(w) << ( left ? left->label : "-" )
       << " r " << std::setw(w) << ( right ? right->label : "-" )
       ;
    return ss.str();
}

bool no::is_primitive() const 
{
    return left == NULL && right == NULL ; 
}
bool no::is_operator() const 
{
    return left != NULL && right != NULL ; 
}
bool no::is_zero() const 
{
    return type == CSG_ZERO ;  
}
bool no::is_lrzero() const 
{
    return is_operator() && left->is_zero() && right->is_zero() ; 
}
bool no::is_rzero() const 
{
    return is_operator() && !left->is_zero() && right->is_zero() ; 
}
bool no::is_lzero() const 
{
    return is_operator() && left->is_zero() && !right->is_zero() ; 
}



no no::make_node(OpticksCSG_t type, no* left, no* right )
{
    no n ;   

    std::string tag = CSGTag(type) ;
    n.label = strdup(tag.c_str()) ;    
    n.left = left ; 
    n.right = right ; 
    n.depth = 0 ; 
    n.type = type ; 
    n.complement = false ; 

    return n ;
}


