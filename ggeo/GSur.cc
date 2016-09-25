#include <sstream>
#include <iomanip>

#include "GSur.hh"
#include "GPropertyMap.hh"

#include "PLOG.hh"

//typedef std::pair<const char*, const char*> PCC ; 
typedef std::pair<std::string, std::string> PSS ; 

GSur::GSur(GPropertyMap<float>* pmap, char type)
    :
    m_pmap(pmap),
    m_type(type),
    m_pv1(NULL),
    m_pv2(NULL),
    m_lv(NULL)
{
}

char GSur::getType()
{
    return m_type ; 
}

const char* GSur::getName()
{
    return m_pmap->getName();
}  


void GSur::setPVP(const char* pv1, const char* pv2)
{
    m_pv1 = strdup(pv1);
    m_pv2 = strdup(pv2);
}
void GSur::setLV(const char* lv)
{
    m_lv = strdup(lv);
}


const char* GSur::getPV1()
{
    return m_pv1 ; 
}
const char* GSur::getPV2()
{
    return m_pv2 ; 
}
const char* GSur::getLV()
{
    return m_lv ; 
}






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
void GSur::addPVPair(const char* pv1, const char* pv2)
{
    m_pvp.insert(PSS(pv1,pv2));
}
void GSur::addLV(const char* lv)
{
    m_slv.insert(lv);
}

const std::set<unsigned>& GSur::getIBnd()
{
    return m_ibnd ; 
}
const std::set<unsigned>& GSur::getOBnd()
{
    return m_obnd ; 
}

unsigned GSur::getNumPVPair()
{
    return m_pvp.size() ;
}
unsigned GSur::getNumLV()
{
    return m_slv.size();
}


void GSur::assignVolumes()
{
    unsigned npvp = getNumPVPair() ;
    unsigned nlv = getNumLV();

    if( npvp == 0 && nlv == 0) m_type = 'U' ;  // change type for unused surfaces

    if(m_type == 'B')
    {
        assert( npvp == 1 );
        std::set<PSS>::const_iterator it=m_pvp.begin() ;
        std::string pv = it->first ; 
        std::string ppv = it->second ; 

        setPVP(pv.c_str(),ppv.c_str());
    }
    else if( m_type == 'S')
    {
        assert( nlv == 1);
        std::set<std::string>::const_iterator it=m_slv.begin() ;
        std::string lv = *it ; 

        setLV(lv.c_str());
    }
}

/*

2016-09-25 15:27:26.307 INFO  [315703] [GSurLib::dump@157] test_GSurLib

    2 B(   2                 NearOWSLinerSurface)  pv1:__dd__Geometry__Pool__lvNearPoolLiner--pvNearPoolOWS0xbf55b10      pv2:__dd__Geometry__Pool__lvNearPoolDead--pvNearPoolLiner0xbf4b270   [ ibnd   14:Tyvek//NearOWSLinerSurface/OwsWater] 
    3 B(   3               NearIWSCurtainSurface)  pv1:__dd__Geometry__Pool__lvNearPoolCurtain--pvNearPoolIWS0xc15a498    pv2:__dd__Geometry__Pool__lvNearPoolOWS--pvNearPoolCurtain0xc5c5f20  [ ibnd   16:Tyvek//NearIWSCurtainSurface/IwsWater] 
    5 B(   5                       SSTOilSurface)  pv1:__dd__Geometry__AD__lvSST--pvOIL0xc241510                          pv2:__dd__Geometry__AD__lvADE--pvSST0xc128d90                        [ ibnd   19:StainlessSteel//SSTOilSurface/MineralOil] 

    1 B(   1                NearDeadLinerSurface)  pv1:__dd__Geometry__Sites__lvNearHallBot--pvNearPoolDead0xc13c018      pv2:__dd__Geometry__Pool__lvNearPoolDead--pvNearPoolLiner0xbf4b270   [ obnd   13:DeadWater/NearDeadLinerSurface//Tyvek] 
    4 B(   4                SSTWaterSurfaceNear1)  pv1:__dd__Geometry__Pool__lvNearPoolIWS--pvNearADE10xc2cf528           pv2:__dd__Geometry__AD__lvADE--pvSST0xc128d90                        [ obnd   18:IwsWater/SSTWaterSurfaceNear1//StainlessSteel] 
    9 B(   9                    ESRAirSurfaceTop)  pv1:__dd__Geometry__AdDetails__lvTopReflector--pvTopRefGap0xc266468    pv2:__dd__Geometry__AdDetails__lvTopRefGap--pvTopESR0xc4110d0        [ obnd   39:Air/ESRAirSurfaceTop//ESR] 
   10 B(  10                    ESRAirSurfaceBot)  pv1:__dd__Geometry__AdDetails__lvBotReflector--pvBotRefGap0xbfa6458    pv2:__dd__Geometry__AdDetails__lvBotRefGap--pvBotESR0xbf9bd08        [ obnd   40:Air/ESRAirSurfaceBot//ESR] 
   12 B(  12                SSTWaterSurfaceNear2)  pv1:__dd__Geometry__Pool__lvNearPoolIWS--pvNearADE20xc0479c8           pv2:__dd__Geometry__AD__lvADE--pvSST0xc128d90                        [ obnd   80:IwsWater/SSTWaterSurfaceNear2//StainlessSteel] 






*/





std::string GSur::brief()
{
    std::stringstream ss ; 
    ss 
        << m_type
        << "( " << std::setw(3) << m_pmap->getIndex()
        << " "  << std::setw(35) << m_pmap->getName()
        << ") "
        ;

    if(m_type == 'B' )
    {
        ss << " pv1:" << m_pv1 ;
        ss << " pv2:" << m_pv2 ;
    } 
    else if(m_type == 'S' )
    {
        ss << " lv:" << m_lv ;
    }
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


void GSur::dump(const char* msg)
{
    LOG(info) << msg  << " " << brief() ;

    for(std::set<PSS>::const_iterator it=m_pvp.begin() ; it != m_pvp.end() ; it++ )
    {
        std::string pv = it->first ; 
        std::string ppv = it->second ; 

        std::cout 
               << " pvp " 
               << std::setw(60) << pv
               << std::setw(60) << ppv
               << std::endl ; 
    }

    for(std::set<std::string>::const_iterator it=m_slv.begin() ; it != m_slv.end() ; it++ )
    {
        std::string lv = *it ; 
        std::cout 
               << " lv  " 
               << std::setw(60) << lv
               << std::endl ; 
    }
}




