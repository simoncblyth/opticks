/*

Q: Where/What uses this ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~



*/


#include <algorithm>
#include <iomanip>

#include "Opticks.hh"

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
    m_ok(gg->getOpticks()),
    m_slib(gg->getSurfaceLib()),
    m_blib(gg->getBndLib()),
    m_closed(false)
{
    init();
}


GSurfaceLib* GSurLib::getSurfaceLib()
{
   return m_slib ; 
}


void GSurLib::pushBorderSurfacesDYB(std::vector<std::string>& names)
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
    if(m_ok->isTest())
    {
    }
    else if(m_ok->isDayabay())
    {
        pushBorderSurfacesDYB(m_bordersurface);
    }
    collectSur();
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

void GSurLib::close()
{
    m_closed = true ; 
    examineSolidBndSurfaces();  
    assignType();
}

bool GSurLib::isClosed()
{
    return m_closed ; 
}



void GSurLib::examineSolidBndSurfaces()
{
    // this is deferred to CDetector::attachSurfaces 
    // to allow CTestDetector to fixup mesh0 info 

    GGeo* gg = m_ggeo ; 

    GMergedMesh* mm = gg->getMergedMesh(0) ;

    unsigned numSolids = mm->getNumSolids();

    LOG(info) << "GSurLib::examineSolidBndSurfaces" 
              << " numSolids " << numSolids
              ; 

    for(unsigned i=0 ; i < numSolids ; i++)
    {
        guint4 id = mm->getIdentity(i);
        guint4 ni = mm->getNodeInfo(i);
        const char* lv = gg->getLVName(i) ;

        // hmm for test geometry the lv returned are the global ones, not the test geometry ones
        // and the boundary names look wrong too

        unsigned node = ni.z ;
        unsigned parent = ni.w ;

        unsigned node2 = id.x ;
        unsigned boundary = id.z ;

        std::string bname = m_blib->shortname(boundary);

        if(node != i)
           LOG(fatal) << "GSurLib::examineSolidBndSurfaces"
                      << " i(mm-idx) " << std::setw(6) << i
                      << " node(ni.z) " << std::setw(6) << node
                      << " node2(id.x) " << std::setw(6) << node2
                      << " boundary(id.z) " << std::setw(6) << boundary
                      << " parent(ni.w) " << std::setw(6) << parent 
                      << " bname " << bname
                      << " lv " << ( lv ? lv : "NULL" )
                      ;

        assert( node == i );


        //unsigned mesh = id.y ;
        //unsigned sensor = id.w ;
        assert( node2 == i );
        
        guint4 bnd = m_blib->getBnd(boundary);

        //unsigned omat_ = bnd.x ; 
        unsigned osur_ = bnd.y ; 
        unsigned isur_ = bnd.z ; 
        //unsigned imat_ = bnd.w ; 


        GSur* isur = isur_ == UNSET ? NULL : getSur(isur_);
        GSur* osur = osur_ == UNSET ? NULL : getSur(osur_);


        LOG(debug) << std::setw(3) << i 
                  << " nodeinfo " << std::setw(50) << ni.description() 
                  << " bnd " << std::setw(50) << bnd.description() 
                  << ( isur ? " isur" : "" )
                  << ( osur ? " osur" : "" )
                  ;


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


