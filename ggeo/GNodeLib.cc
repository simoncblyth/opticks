#include <sstream>

#include "PLOG.hh"

#include "Opticks.hh"

#include "GItemList.hh"
#include "GSolid.hh"
#include "GNodeLib.hh"
#include "GTreePresent.hh"


const char* GNodeLib::GetRelDir(bool analytic, bool test)
{
    std::stringstream ss ; 
    ss << "GNodeLib" ; 
    if(analytic) ss << "Analytic" ; 
    if(test)     ss << "Test" ; 

    std::string s = ss.str() ;
    return strdup(s.c_str()) ;
}

GNodeLib* GNodeLib::Load(Opticks* ok, bool analytic, bool test)
{
    GNodeLib* nodelib = new GNodeLib(ok, analytic, test) ;
    nodelib->loadFromCache();
    return nodelib ; 
}

void GNodeLib::loadFromCache()
{
    const char* idpath = m_ok->getIdPath() ;
    m_pvlist = GItemList::load(idpath, "PVNames", m_reldir);
    m_lvlist = GItemList::load(idpath, "LVNames", m_reldir);
}

GNodeLib::GNodeLib(Opticks* ok, bool analytic, bool test)  
    :
    m_ok(ok),
    m_analytic(analytic),
    m_test(test),
    m_reldir(GetRelDir(analytic,test)),
    m_pvlist(NULL),
    m_lvlist(NULL),
    m_treepresent(new GTreePresent(100, 1000))   // depth_max,sibling_max
{
}


void GNodeLib::save() const 
{
    const char* idpath = m_ok->getIdPath() ;
    LOG(debug) << "GNodeLib::save"
              << " idpath " << idpath 
              ;

    if(m_pvlist)
    {
        m_pvlist->save(idpath);
    }
    else
    {
        LOG(warning) << "GNodeLib::save pvlist NULL " ; 
    }


    if(m_lvlist)
    {
        m_lvlist->save(idpath);
    }
    else
    {
        LOG(warning) << "GNodeLib::save lvlist NULL " ; 
    }


    GNode* top = getNode(0); 
    m_treepresent->traverse(top);
    m_treepresent->write(idpath, m_reldir);
}


std::string GNodeLib::desc() const 
{
    std::stringstream ss ; 

    ss << "GNodeLib"
       << " reldir " << ( m_reldir ? m_reldir : "-" )
       << " numPV " << getNumPV()
       << " numLV " << getNumLV()
       << " numSolids " << getNumSolids()
       << " PV(0) " << getPVName(0)
       << " LV(0) " << getLVName(0)
       ;


    typedef std::map<unsigned, GSolid*>::const_iterator IT ; 

    IT beg = m_solidmap.begin() ;
    IT end = m_solidmap.end() ;

    for(IT it=beg ; it != end && std::distance(beg,it) < 10 ; it++)
    {
        ss << " ( " << it->first << " )" ; 
    }

    return ss.str();
}



unsigned GNodeLib::getNumPV() const 
{
    unsigned npv = m_pvlist->getNumKeys(); 
    return npv ; 
}

unsigned GNodeLib::getNumLV() const 
{
    unsigned nlv = m_lvlist->getNumKeys(); 
    return nlv ; 
}

const char* GNodeLib::getPVName(unsigned int index) const 
{
    return m_pvlist ? m_pvlist->getKey(index) : NULL ; 
}
const char* GNodeLib::getLVName(unsigned int index) const 
{
    return m_lvlist ? m_lvlist->getKey(index) : NULL ; 
}


unsigned int GNodeLib::getNumSolids() const 
{
    return m_solids.size();
}


GItemList* GNodeLib::getPVList()
{
    return m_pvlist ; 
}
GItemList* GNodeLib::getLVList()
{
    return m_lvlist ; 
}



void GNodeLib::add(GSolid* solid)
{
    m_solids.push_back(solid);

    unsigned int index = solid->getIndex(); 

    if(m_test)
    {
        assert( m_solids.size() - 1 == index && "indices of test geometry solids added to GNodeLib must follow the sequence : 0,1,2,... " );
    }


    //assert( m_solidmap.size() == index );   //  only with relative GSolid indexing

/*
    LOG(info) << "GNodeLib::add"
              << " solidIndex " << index 
              << " preCount " << m_solidmap.size()
              ;
*/
              
    m_solidmap[index] = solid ; 

    if(!m_pvlist) m_pvlist = new GItemList("PVNames", m_reldir) ; 
    if(!m_lvlist) m_lvlist = new GItemList("LVNames", m_reldir) ; 

    m_lvlist->add(solid->getLVName()); 
    m_pvlist->add(solid->getPVName()); 

    // NB added in tandem, so same counts and same index as the solids  

    GSolid* check = getSolid(index);
    assert(check == solid);
}


GSolid* GNodeLib::getSolid(unsigned index) const 
{
    GSolid* solid = NULL ; 
    if(m_solidmap.find(index) != m_solidmap.end()) 
    {
        solid = m_solidmap.at(index) ;
        assert(solid->getIndex() == index);
    }
    return solid ; 
}

GSolid* GNodeLib::getSolidSimple(unsigned int index)
{
    return m_solids[index];
}


GNode* GNodeLib::getNode(unsigned index) const 
{
    GSolid* solid = getSolid(index);
    GNode* node = static_cast<GNode*>(solid); 
    return node ; 
}



