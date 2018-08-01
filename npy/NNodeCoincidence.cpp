#include <sstream>
#include <iomanip>
#include "NNode.hpp"
#include "NNodeCoincidence.hpp"

bool NNodeCoincidence::is_siblings() const
{
   return i->parent && j->parent && i->parent == j->parent ;
}

bool NNodeCoincidence::is_union_siblings() const
{
   return is_siblings() && i->parent->type == CSG_UNION ;
}

bool NNodeCoincidence::is_union_parents() const
{
   return i->parent && j->parent && i->parent->type == CSG_UNION && j->parent->type == CSG_UNION ;
}

std::string NNodeCoincidence::desc() const 
{
    std::stringstream ss ; 
    ss
        << "(" << std::setw(2) << i->idx
        << "," << std::setw(2) << j->idx
        << ")"
        << " " << NNodeEnum::PairType(p)
        << " " << NNodeEnum::NudgeType(n)
        << " " << i->tag()
        << " " << j->tag()
        << " sibs " << ( is_siblings() ? "Y" : "N" )
        << " u_sibs " << ( is_union_siblings() ? "Y" : "N" )
        << " u_par " << ( is_union_parents() ? "Y" : "N" )
        << " u_same " << ( nnode::is_same_union(i,j) ? "Y" : "N" )
        << " " << ( fixed ? "FIXED" : "" )
        ; 
  
    return ss.str();
}


