#include "NPYSpec.hpp"
#include "NLog.hpp"
#include <cstring>
#include <cstdio>

void NPYSpec::Summary(const char* msg) 
{
        printf("%s : %10u %10u %10u %10u \n", msg, m_ni, m_nj, m_nk, m_nl);
}

std::string NPYSpec::description() 
{
     char s[64] ;
     snprintf(s, 64, " (%3u,%3u,%3u,%3u) ", m_ni, m_nj, m_nk, m_nl);
     return s ;
}

const char* NPYSpec::getTypeName()
{
    return NPYBase::TypeName(m_type);
}

unsigned int NPYSpec::getDimension(unsigned int i) 
{
    switch(i)
    {
        case 0:return m_ni; break;
        case 1:return m_nj; break;
        case 2:return m_nk; break;
        case 3:return m_nl; break;
    }
    return m_bad_index ; 
}

bool NPYSpec::isEqualTo(NPYSpec* other) 
{
    bool match = 
         getDimension(0) == other->getDimension(0) &&
         getDimension(1) == other->getDimension(1) &&
         getDimension(2) == other->getDimension(2) &&
         getDimension(3) == other->getDimension(3) &&
         getType() == other->getType()
         ;


    if(!match)
    {

       for(int i=0 ; i < 4 ; i++)
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







