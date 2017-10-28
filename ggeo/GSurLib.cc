// See notes/issues/surface_review.rst


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

GSurfaceLib* GSurLib::getSurfaceLib()
{
   return m_slib ; 
}

Opticks* GSurLib::getOpticks()
{
   return m_ok ;  
}



GSurLib::GSurLib(GGeo* gg) 
    : 
    m_ggeo(gg),
    m_ok(gg->getOpticks()),
    m_dbgsurf(m_ok->isDbgSurf()),
    m_slib(gg->getSurfaceLib()),
    m_blib(gg->getBndLib()),
    m_closed(false)
{
    init();
}


void GSurLib::init()
{
    if(m_dbgsurf) LOG(info) << "[--dbgsurf] GSurLib::init" ; 
    if(m_ok->isDayabay())
    {
        pushBorderSurfacesDYB(m_bordersurface);
        if(m_dbgsurf) 
           LOG(info) << "[--dbgsurf] GSurLib::init m_bordersurface.size " << m_bordersurface.size() ; 
    }

    // even with test geometries, still want to have access to the 
    // cheat surface classifications in the basis geometry 

    collectSur();
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


void GSurLib::collectSur()
{
    unsigned nsur = m_slib->getNumSurfaces();

    if(m_dbgsurf) 
         LOG(info) << "[--dbgsurf]" 
                   << " m_slib numSurfaces " << nsur
                   ;


    for(unsigned i=0 ; i < nsur ; i++)
    {
        GPropertyMap<float>* pm = m_slib->getSurface(i);  
        const char* name = pm->getName();
        char type = isBorderSurface(name) ? 'B' : 'S' ;   // border or skin surface 

        if(m_dbgsurf)
            LOG(info) << "[--dbgsurf]"
                      << " i " << std::setw(3) << i 
                      << " type " << type
                      << " name " << name
                      ;

        GSur* sur = new GSur(pm, type);
        add(sur);
    }
}


void GSurLib::close()
{
    if(m_dbgsurf) 
         LOG(info) << "[--dbgsurf] GSurLib::close START " ;

    m_closed = true ; 
    examineSolidBndSurfaces();  
    assignType();

    if(m_dbgsurf) 
         LOG(info) << "[--dbgsurf] GSurLib::close DONE " ;
}

bool GSurLib::isClosed()
{
    return m_closed ; 
}


void GSurLib::getSurfacePair(std::pair<GSur*,GSur*>& osur_isur, unsigned boundary)
{
    guint4 bnd = m_blib->getBnd(boundary);

    if(m_dbgsurf)
        LOG(info) << " GSurLib::getSurfacePair "
                  << " bnd " << std::setw(50) << bnd.description() 
                  ;

    unsigned osur_ = bnd.y ; 
    unsigned isur_ = bnd.z ; 

    GSur* isur = isur_ == UNSET ? NULL : getSur(isur_);
    GSur* osur = osur_ == UNSET ? NULL : getSur(osur_);
 
    osur_isur.first  = osur ; 
    osur_isur.second = isur ; 
}



void GSurLib::examineSolidBndSurfaces()
{
    // this is deferred to CDetector::attachSurfaces 
    // to allow CTestDetector to fixup mesh0 info 
    //
    // hmm GGeoTest(NCSG) has another mm 
    // even though the polygonization is often not good
    // that doesnt prevent the below from being able to work.

    GGeo* gg = m_ggeo ; 

    GMergedMesh* mm = gg->getMergedMesh(0) ;
    unsigned numSolids = mm->getNumSolids();

    if(m_dbgsurf)
    LOG(info) << "[--dbgsurf] GSurLib::examineSolidBndSurfaces" 
              << " numSolids " << numSolids
              << " mm " << mm 
              ; 


    // lookup surface pair for each boundary 
    // and invoke methods that will allow 
    // G4 surface creation to find the appropriate pv and lv arguments 

    unsigned node_mismatch(0);
    unsigned node2_mismatch(0);

    // hmm for test geometry the lv returned are the global ones, not the test geometry ones
    // and the boundary names look wrong too
    //
    // hmm for test geometry where are creating the 
    // lv, pv it would be simpler to do this during creation rather than 
    // afterwards

    bool reverse = true ; 

    for(unsigned j=0 ; j < numSolids ; j++)
    {
        unsigned i = reverse ? numSolids - 1 - j : j ; 

        const char* lv = gg->getLVName(i) ;

        guint4 identity = mm->getIdentity(i);
        unsigned node2 = identity.x ;
        unsigned boundary = identity.z ;

        guint4 nodeinfo = mm->getNodeInfo(i);
        unsigned node = nodeinfo.z ;
        unsigned parent = nodeinfo.w ;

        std::string bname = m_blib->shortname(boundary);

        // j not i, as order reversal elsewhere (for auto-containment ?)
        if( node != j ) node_mismatch++ ;
        if( node2 != j ) node2_mismatch++ ;
        
        std::pair<GSur*,GSur*> osur_isur ; 
        getSurfacePair(osur_isur, boundary );

        GSur* osur = osur_isur.first ; 
        GSur* isur = osur_isur.second ; 

        if(m_dbgsurf)
           LOG(info) << "GSurLib::examineSolidBndSurfaces"
                      << " j " << std::setw(6) << j
                      << " i(so-idx) " << std::setw(6) << i
                      << " node(ni.z) " << std::setw(6) << node
                      << " node2(id.x) " << std::setw(6) << node2
                      << " boundary(id.z) " << std::setw(6) << boundary
                      << " parent(ni.w) " << std::setw(6) << parent 
                      << " nodeinfo " << std::setw(50) << nodeinfo.description() 
                      << " bname " << bname
                      << " lv " << ( lv ? lv : "NULL" )
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

    LOG(info) 
       << " node_mismatch " << node_mismatch
       << " node2_mismatch " << node2_mismatch
       ;

    assert( node_mismatch == 0 );
    assert( node2_mismatch == 0 );
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


