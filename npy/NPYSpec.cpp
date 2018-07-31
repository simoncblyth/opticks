#include "NPYSpec.hpp"

#include <sstream>
#include <cstring>
#include <cstdio>
#include <climits>
#include <cassert>

#include "PLOG.hh"

NPYSpec::NPYSpec(const char* name, unsigned ni, unsigned nj, unsigned nk, unsigned nl, unsigned nm, NPYBase::Type_t type, const char* ctrl, bool optional)
  :
    m_name(name ? strdup(name) : NULL),
    m_ni(ni),
    m_nj(nj),
    m_nk(nk),
    m_nl(nl),
    m_nm(nm),
    m_bad_index(UINT_MAX), 
    m_type(type),
    m_ctrl(ctrl ? strdup(ctrl) : NULL),
    m_optional(optional)
{
}

NPYSpec* NPYSpec::clone() const 
{
    return new NPYSpec(m_name, m_ni, m_nj, m_nk, m_nl, m_nm, m_type, m_ctrl, m_optional );
}
void NPYSpec::setNumItems(unsigned ni)
{
    m_ni = ni ; 
}

NPYBase::Type_t NPYSpec::getType() const 
{ 
    return m_type ;
}

const char* NPYSpec::getCtrl() const 
{
    return m_ctrl ; 
}

bool NPYSpec::isOptional() const 
{
    return m_optional ; 
}
const char* NPYSpec::getName() const 
{
    return m_name ; 
}

void NPYSpec::Summary(const char* msg) const
{
    printf("%s : %20s %10u %10u %10u %10u %10u \n", msg, (m_name ? m_name : ""), m_ni, m_nj, m_nk, m_nl, m_nm);
}

std::string NPYSpec::description() const  
{
     std::stringstream ss ; 
     ss << std::setw(20) << desc()
        << " " 
        << std::setw(3) << ( m_optional ? "O" : " " )
        << std::setw(20) << ( m_name ? m_name : "-" )
        << std::setw(20) << getTypeName()
        ; 
     return ss.str(); 
} 


std::string NPYSpec::desc() const  
{
     char s[64] ;
     snprintf(s, 64, " (%3u,%3u,%3u,%3u,%3u) ", m_ni, m_nj, m_nk, m_nl, m_nm);
     return s ;
}

const char* NPYSpec::getTypeName() const 
{
    return NPYBase::TypeName(m_type);
}

unsigned int NPYSpec::getDimension(unsigned int i) const 
{
    switch(i)
    {
        case 0:return m_ni; break;
        case 1:return m_nj; break;
        case 2:return m_nk; break;
        case 3:return m_nl; break;
        case 4:return m_nm; break;
    }
    assert(0); 
    return m_bad_index ; 
}

bool NPYSpec::isEqualTo(const NPYSpec* other) const
{
    bool match = 
         getDimension(0) == other->getDimension(0) &&
         getDimension(1) == other->getDimension(1) &&
         getDimension(2) == other->getDimension(2) &&
         getDimension(3) == other->getDimension(3) &&
         getDimension(4) == other->getDimension(4) &&
         getType() == other->getType()
         ;

    if(!match)
    {

       for(int i=0 ; i < 5 ; i++)
          LOG(info) << "NPYSpec::isEqualTo" 
                    << " i " << i 
                    << " self " << getDimension(i)
                    << " other " << other->getDimension(i)
                    << " match? " << ( getDimension(i) == other->getDimension(i) )
                    ;
 
        LOG(info) << "NPYSpec::isEqualTo"
                  << " type " << getType()
                  << " typeName " << getTypeName()
                  << " other type " << other->getType()
                  << " other typeName " << other->getTypeName()
                  << " match? " << ( getType() == other->getType() )
                  ;

    }

    return match ; 
}


