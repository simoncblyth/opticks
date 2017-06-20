#include "PLOG.hh"

#include "Opticks.hh"

#include "GItemList.hh"
#include "GSolid.hh"
#include "GNodeLib.hh"


GNodeLib* GNodeLib::load(Opticks* ok)
{
    GNodeLib* nodelib = new GNodeLib(ok, true) ;
    return nodelib ; 
}

GNodeLib::GNodeLib(Opticks* ok, bool loaded )  
    :
    m_ok(ok),
    m_loaded(loaded),
    m_pvlist(NULL),
    m_lvlist(NULL)
{
    init();
}



void GNodeLib::init()
{
    if(!m_loaded)
    {
        m_pvlist = new GItemList("GNodeLib_PVNames") ; 
        m_lvlist = new GItemList("GNodeLib_LVNames") ; 
    }
    else
    {
        const char* idpath = m_ok->getIdPath() ;
        m_pvlist = GItemList::load(idpath, "GNodeLib_PVNames");
        m_lvlist = GItemList::load(idpath, "GNodeLib_LVNames");
    }
}

void GNodeLib::save()
{
    const char* idpath = m_ok->getIdPath() ;
    LOG(info) << "GNodeLib::save"
              << " idpath " << idpath 
              ;
    m_pvlist->save(idpath);
    m_lvlist->save(idpath);
}


unsigned GNodeLib::getNumPV()
{
    unsigned npv = m_pvlist->getNumKeys(); 
    return npv ; 
}

unsigned GNodeLib::getNumLV()
{
    unsigned nlv = m_lvlist->getNumKeys(); 
    return nlv ; 
}

const char* GNodeLib::getPVName(unsigned int index)
{
    return m_pvlist ? m_pvlist->getKey(index) : NULL ; 
}
const char* GNodeLib::getLVName(unsigned int index)
{
    return m_lvlist ? m_lvlist->getKey(index) : NULL ; 
}


unsigned int GNodeLib::getNumSolids()
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

    unsigned int index = solid->getIndex(); // absolute node index, independent of the selection

    LOG(info) << "GNodeLib::add"
              << " solidIndex " << index 
              << " preCount " << m_solidmap.size()
              ;
              
    m_solidmap[index] = solid ; 

    m_lvlist->add(solid->getLVName()); 
    m_pvlist->add(solid->getPVName()); 

    // NB added in tandem, so same counts and same index as the solids  

    GSolid* check = getSolid(index);
    assert(check == solid);
}


GSolid* GNodeLib::getSolid(unsigned index)
{
    GSolid* solid = NULL ; 
    if(m_solidmap.find(index) != m_solidmap.end()) 
    {
        solid = m_solidmap[index] ;
        assert(solid->getIndex() == index);
    }
    return solid ; 
}

GSolid* GNodeLib::getSolidSimple(unsigned int index)
{
    return m_solids[index];
}


GNode* GNodeLib::getNode(unsigned index)
{
    GSolid* solid = getSolid(index);
    GNode* node = static_cast<GNode*>(solid); 
    return node ; 
}



