#include <cstdio>
#include <climits>
#include <cstring>


// npy-
#include "NGLM.hpp"
#include "NSensor.hpp"

#include "GPropertyMap.hh"
#include "GMesh.hh"

#include "GMatrix.hh"
#include "GBndLib.hh"
#include "GSurfaceLib.hh"

#include "GSolid.hh"

#include "PLOG.hh"
#include "GGEO_BODY.hh"


GSolid::GSolid( unsigned int index, GMatrix<float>* transform, GMesh* mesh, unsigned int boundary, NSensor* sensor)
         : 
         GNode(index, transform, mesh ),
         m_boundary(boundary),
         m_csgflag(CSG_PARTLIST),
         m_sensor(sensor),
         m_selected(true),
         m_pvname(NULL),
         m_lvname(NULL),
         m_sensor_surface_index(0)
{
}


OpticksCSG_t GSolid::getCSGFlag()
{
    return m_csgflag ; 
}

unsigned int GSolid::getBoundary()
{
    return m_boundary ; 
}

NSensor* GSolid::getSensor()
{
    return m_sensor ; 
}

void GSolid::setSelected(bool selected)
{
    m_selected = selected ; 
}
bool GSolid::isSelected()
{
   return m_selected ; 
}

void GSolid::setPVName(const char* pvname)
{
    m_pvname = strdup(pvname);
}
void GSolid::setLVName(const char* lvname)
{
    m_lvname = strdup(lvname);
}

const char* GSolid::getPVName()
{
    return m_pvname ; 
}
const char* GSolid::getLVName()
{
    return m_lvname ; 
}

void GSolid::setSensorSurfaceIndex(unsigned int ssi)
{
    m_sensor_surface_index = ssi ; 
}
unsigned int GSolid::getSensorSurfaceIndex()
{
    return m_sensor_surface_index ; 
}









void GSolid::Summary(const char* msg )
{
   if(!msg) msg = getDescription();
   if(!msg) msg = "GSolid::Summary" ;
   printf("%s\n", msg );
}


std::string GSolid::description()
{
   return getDescription(); 
}

GParts* GSolid::getParts()
{
    return m_mesh ? m_mesh->getParts() : NULL ; 
}
void GSolid::setParts(GParts* pts)
{
    if(m_mesh) m_mesh->setParts(pts) ;
}



void GSolid::setCSGFlag(OpticksCSG_t csgflag)
{
    m_csgflag = csgflag ; 
}

void GSolid::setBoundary(unsigned int boundary)
{
    m_boundary = boundary ; 
    setBoundaryIndices( boundary );
}


void GSolid::setBoundaryAll(unsigned boundary)
{
     unsigned nchild = getNumChildren();
     if(nchild > 0)
     {
        for(unsigned i=0 ; i < nchild ; i++)
        {
            GNode* node = getChild(i);
            GSolid* sub = dynamic_cast<GSolid*>(node);
            sub->setBoundary(boundary);
        }
     } 
 }





void GSolid::setSensor(NSensor* sensor)
{
    m_sensor = sensor ; 
    // every triangle needs a value... use 0 to mean unset, so sensor   
    setSensorIndices( NSensor::RefIndex(sensor) );
}

guint4 GSolid::getIdentity()
{
    return guint4(
                   m_index, 
                   m_mesh ? m_mesh->getIndex() : 0, 
                   m_boundary,
                   getSensorSurfaceIndex()
                 );
}
 


void GSolid::Dump( const std::vector<GSolid*>& solids, const char* msg )
{
    unsigned numSolid = solids.size() ;
    LOG(info) << msg << " numSolid " << numSolid ; 
    for(unsigned i=0 ; i < numSolid ; i++) solids[i]->dump(); 
}


