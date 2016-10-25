#include <algorithm>
#include <iomanip>

#include "GGeo.hh"
#include "GMergedMesh.hh"
#include "GSurfaceLib.hh"
#include "GPropertyMap.hh"
#include "GBndLib.hh"

#include "GSurLib.hh"
#include "GSur.hh"

#include "PLOG.hh"

const unsigned GSurLib::UNSET = -1 ; 


void GSurLib::add(GSur* sur)
{
    m_surs.push_back(sur);
}
unsigned GSurLib::getNumSur()
{
    return m_surs.size();
}
GSur* GSurLib::getSur(unsigned i)
{
    unsigned numSur = getNumSur();
    return i < numSur ? m_surs[i] : NULL ; 
}



GSurLib::GSurLib(GGeo* gg) 
    : 
    m_ggeo(gg),
    m_mesh0(gg->getMergedMesh(0)),
    m_slib(gg->getSurfaceLib()),
    m_blib(gg->getBndLib())
{
    init();
}


GSurfaceLib* GSurLib::getSurfaceLib()
{
   return m_slib ; 
}


void GSurLib::pushBorderSurfaces(std::vector<std::string>& names)
{
    // cheat determination of bordersurfaces by looking at the .dae
    // these have directionality, the other **skin** surfaces do not
    names.push_back("ESRAirSurfaceTop");
    names.push_back("ESRAirSurfaceBot");
    names.push_back("SSTOilSurface");
    names.push_back("SSTWaterSurfaceNear1");
    names.push_back("SSTWaterSurfaceNear2");
    names.push_back("NearIWSCurtainSurface");
    names.push_back("NearOWSLinerSurface");
    names.push_back("NearDeadLinerSurface");
}

bool GSurLib::isBorderSurface(const char* name)
{
    return std::find(m_bordersurface.begin(), m_bordersurface.end(), name ) != m_bordersurface.end() ; 
}

void GSurLib::init()
{
    pushBorderSurfaces(m_bordersurface);
    collectSur();
    examineSolidBndSurfaces();  
    assignType();
}

void GSurLib::collectSur()
{
    unsigned nsur = m_slib->getNumSurfaces();
    LOG(info) << " nsur " << nsur ; 
    for(unsigned i=0 ; i < nsur ; i++)
    {
        GPropertyMap<float>* pm = m_slib->getSurface(i);  
        const char* name = pm->getName();
        char type = isBorderSurface(name) ? 'B' : 'S' ;   // border or skin surface 
        GSur* sur = new GSur(pm, type);
        add(sur);
    }
}

void GSurLib::examineSolidBndSurfaces()
{
    GGeo* gg = m_ggeo ; 
    GMergedMesh* mm = m_mesh0 ; 

    unsigned numSolids = mm->getNumSolids();

    for(unsigned i=0 ; i < numSolids ; i++)
    {
        guint4 nodeinfo = mm->getNodeInfo(i);
        unsigned node = nodeinfo.z ;
        unsigned parent = nodeinfo.w ;
        assert( node == i );

        guint4 id = mm->getIdentity(i);
        unsigned node2 = id.x ;
        //unsigned mesh = id.y ;
        unsigned boundary = id.z ;
        //unsigned sensor = id.w ;
        assert( node2 == i );
        
        std::string bname = m_blib->shortname(boundary);
        guint4 bnd = m_blib->getBnd(boundary);

        //unsigned omat_ = bnd.x ; 
        unsigned osur_ = bnd.y ; 
        unsigned isur_ = bnd.z ; 
        //unsigned imat_ = bnd.w ; 

        const char* lv = gg->getLVName(i) ;

        GSur* isur = isur_ == UNSET ? NULL : getSur(isur_);
        GSur* osur = osur_ == UNSET ? NULL : getSur(osur_);

        // the reason for the order swap between osur and isur is explained below
        if(osur)
        {
            osur->addOuter(i, boundary);
            osur->addVolumePair(parent,  i);   // border surface identity based on PV instances, so must use the index NOT THE NAME
            osur->addLV(lv);                   // skin surface identity is 1-to-1 with lv name ?
        } 
        if(isur)
        {
            isur->addInner(i,boundary);
            isur->addVolumePair(i, parent);   // note volume order for isur is opposite to osur
            isur->addLV(lv);
        } 
    }
} 

/**
Boundaries are composed of 4 ordered indices: omat/osur/isur/imat 

* outer and inner material (omat,imat) are mandatory 
* outer and inner surface (osur,isur) are optional
       
The more common "osur" outer surfaces are relevant 
to incoming photons propagating from containing 
parent volume (pv1) into self volume (pv2).

Examples of osur::

      DeadWater/NearDeadLinerSurface//Tyvek
      IwsWater/SSTWaterSurfaceNear1//StainlessSteel
      Air/ESRAirSurfaceTop//ESR
      Air/ESRAirSurfaceBot//ESR
      IwsWater/SSTWaterSurfaceNear2//StainlessSteel
 
The less common "isur" inner surfaces are relevant
to outgoing photons propagating from self volume (pv1) to 
containing parent volume (pv2).

Examples of isur::

     Tyvek//NearOWSLinerSurface/OwsWater
     Tyvek//NearIWSCurtainSurface/IwsWater
     StainlessSteel//SSTOilSurface/MineralOil
  
**/

void GSurLib::assignType()
{
    unsigned numSur = getNumSur();
    for(unsigned i=0 ; i < numSur ; i++)
    {
        GSur* sur = getSur(i);
        sur->assignType();
    }
}

std::string GSurLib::desc(const std::set<unsigned>& bnd)
{
    std::stringstream ss ;
    for(std::set<unsigned>::const_iterator it=bnd.begin() ; it != bnd.end() ; it++)
    {
        unsigned boundary = *it ; 
        ss << " " << std::setw(3) << boundary << ":" << m_blib->shortname(boundary); 
    }
    return ss.str(); 
}


void GSurLib::dump(const char* msg)
{
    LOG(info) << msg ; 

    unsigned numSur = getNumSur();
    for(unsigned i=0 ; i < numSur ; i++)
    {
        GSur* sur = getSur(i);

        //char type = sur->getType();
        //if(type == 'U' || type == 'S' ) continue ; 

        const std::set<unsigned>& ibnd = sur->getIBnd();
        const std::set<unsigned>& obnd = sur->getOBnd();
       
        std::cout
               << std::setw(5) << i 
               << " " << sur->brief() 
               ;

        if(ibnd.size() > 0) std::cout << " [ ibnd " << desc(ibnd) << "] " ;
        if(obnd.size() > 0) std::cout << " [ obnd " << desc(obnd) << "] " ;
 
        std::cout << std::endl ;

        //sur->dump();
    }
}


