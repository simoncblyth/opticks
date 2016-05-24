#include "CDetector.hh"

#include <cstdio>

// npy-
#include "NLog.hpp"
#include "GLMFormat.hpp"

// cfg4-
#include "CPropLib.hh"
#include "CTraverser.hh"

// ggeo-
#include "GCache.hh"


// g4-
#include "G4PVPlacement.hh"


void CDetector::init()
{
    m_lib = new CPropLib(m_cache);
}

void CDetector::traverse(G4VPhysicalVolume* top)
{
    m_traverser = new CTraverser(top); 
    m_traverser->Traverse();
    m_traverser->Summary("CDetector::traverse");
}

void CDetector::saveTransforms(const char* path)
{
    assert(m_traverser);
    m_traverser->saveTransforms(path);
}

unsigned int CDetector::getNumGlobalTransforms()
{
    assert(m_traverser);
    return m_traverser->getNumGlobalTransforms();
}
unsigned int CDetector::getNumLocalTransforms()
{
    assert(m_traverser);
    return m_traverser->getNumLocalTransforms();
}

glm::mat4 CDetector::getGlobalTransform(unsigned int index)
{
    assert(m_traverser);
    return m_traverser->getGlobalTransform(index);
}
glm::mat4 CDetector::getLocalTransform(unsigned int index)
{
    assert(m_traverser);
    return m_traverser->getLocalTransform(index);
}
const char* CDetector::getPVName(unsigned int index)
{
    assert(m_traverser);
    return m_traverser->getPVName(index);
}


void CDetector::dumpPV(const char* msg)
{
    LOG(info) << msg ; 

    typedef std::map<std::string, G4VPhysicalVolume*> MSV ; 

    for(MSV::const_iterator it=m_pvm.begin() ; it != m_pvm.end() ; it++)
    {
         std::string pvn = it->first ; 
         G4VPhysicalVolume* pv = it->second ;  

         std::cout << std::setw(40) << pvn 
                   << std::setw(40) << pv->GetName() 
                   << std::endl 
                   ;

    }
}

G4VPhysicalVolume* CDetector::getPV(const char* name)
{
    return m_pvm.count(name) == 1 ? m_pvm[name] : NULL ; 
}

CDetector::~CDetector()
{
    //printf("CDetector::~CDetector\n");
    //G4GeometryManager::GetInstance()->OpenGeometry();
    //printf("CDetector::~CDetector DONE\n");
}
