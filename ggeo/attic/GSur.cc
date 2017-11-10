#include <sstream>
#include <iomanip>
#include <iterator>  

#include "GSur.hh"
#include "GPropertyMap.hh"
#include "GVector.hh"

#include "PLOG.hh"

typedef std::pair<std::string, std::string> PSS ; 
typedef std::pair<unsigned, unsigned> PUU ; 

GSur::GSur(GPropertyMap<float>* pmap, char type)
    :
    m_pmap(pmap),
    m_type(type)
{
}
char GSur::getType()
{
    return m_type ; 
}
void GSur::setType(char type)
{
    m_type = type ; 
}
const char* GSur::getName()
{
    return m_pmap->getName();
}  
GPropertyMap<float>* GSur::getPMap()
{
    return m_pmap ; 
}




// adders invoked by GSurLib::examineSolidBndSurfaces

void GSur::addInner(unsigned vol, unsigned bnd)
{
    m_ivol.push_back(vol); 
    m_ibnd.insert(bnd);
}
void GSur::addOuter(unsigned vol, unsigned bnd)
{
    m_ovol.push_back(vol); 
    m_obnd.insert(bnd);
}

void GSur::addVolumePair(unsigned pv1, unsigned pv2)
{
    m_pvp.insert(PUU(pv1,pv2));
}
unsigned GSur::getNumVolumePair()
{
    return m_pvp.size() ;
}
guint4 GSur::getVolumePair(unsigned index )
{
    std::set<PUU>::const_iterator iuu=m_pvp.begin() ;
    std::advance(iuu, index);
    assert( iuu != m_pvp.end() ); 

    unsigned pv1 = iuu->first ; 
    unsigned pv2 = iuu->second ; 

    guint4 pair ;

    pair.x = pv1 ; 
    pair.y = pv2 ; 
    pair.z = 0u ; 
    pair.w = index ;

    return pair ;
} 




void GSur::addLV(const char* lv)
{
    m_slv.insert(lv);
}
unsigned GSur::getNumLV()
{
    return m_slv.size();
}

const char* GSur::getLV(unsigned index)
{
    std::set<std::string>::const_iterator is=m_slv.begin() ;
    std::advance(is, index);
    assert( is != m_slv.end() ); 

    std::string lv = *is ; 
    return strdup(lv.c_str());
} 



const std::set<unsigned>& GSur::getIBnd()
{
    return m_ibnd ; 
}
const std::set<unsigned>& GSur::getOBnd()
{
    return m_obnd ; 
}



void GSur::setBorder()
{
    setType('B');
}
void GSur::setSkin()
{
    setType('S'); 
}
void GSur::setUnused()
{
    setType('U') ; 
}


bool GSur::isBorder()
{
    return m_type == 'B' ;
}
bool GSur::isSkin()
{
    return m_type == 'S' ;
}
bool GSur::isUnused()
{
    return m_type == 'U' ;
}


void GSur::assignType()
{
    unsigned nvp = getNumVolumePair() ;
    unsigned nlv = getNumLV();
    if( nvp == 0 && nlv == 0) m_type = 'U' ;  // change type for unused surfaces
}



std::string GSur::brief()
{
    std::stringstream ss ; 
    ss 
        << m_type
        << "( " << std::setw(3) << m_pmap->getIndex()
        << " "  << std::setw(35) << m_pmap->getName()
        << ") "
        << " nlv "  << std::setw(3) << m_slv.size()
        << " npvp " << std::setw(3) << m_pvp.size()
        << " "
        ;

    return ss.str();
}
 
std::string GSur::check()
{
    std::stringstream ss ; 
    ss 
        << m_type
        << "( " << std::setw(3) << m_pmap->getIndex()
        << " "  << std::setw(35) << m_pmap->getName()
        << ") "
        << " ibnd " << std::setw(3) << m_ibnd.size()
        << " obnd " << std::setw(3) << m_obnd.size()
        << " ivol " << std::setw(3) << m_ivol.size()
        << " ovol " << std::setw(3) << m_ovol.size()
        << " npvp " << std::setw(3) << m_pvp.size()
        << " nlv "  << std::setw(3) << m_slv.size()
        ;

    return ss.str();
}


std::string GSur::pvpBrief()
{
    std::stringstream ss ;
    unsigned npvp = m_pvp.size() ; 
    std::set<PUU>::const_iterator it=m_pvp.begin() ;

    for(unsigned i=0 ; i < std::min(npvp, 5u) ; i++)
    {
        std::advance(it, i);
        ss << "(" << it->first << "," << it->second << ")" ; 
    }
    return ss.str();
}


std::string GSur::lvBrief()
{
    std::stringstream ss ;
    for(std::set<std::string>::const_iterator it=m_slv.begin() ; it != m_slv.end() ; it++ )
    {
        std::string lv = *it ; 
        ss  
               << " lv   " 
               << std::setw(60) << lv
               << std::endl ; 
    }
    return ss.str();
}


void GSur::dump(const char* msg)
{
    LOG(info) << msg  << " " 
              << brief() 
              << pvpBrief() 
              << lvBrief()
              ;

}

