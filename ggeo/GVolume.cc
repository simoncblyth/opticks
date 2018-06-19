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

#include "GVolume.hh"

#include "PLOG.hh"
#include "GGEO_BODY.hh"


GVolume::GVolume( unsigned int index, GMatrix<float>* transform, const GMesh* mesh, unsigned int boundary, NSensor* sensor)
         : 
         GNode(index, transform, mesh ),
         m_boundary(boundary),
         m_csgflag(CSG_PARTLIST),
         m_csgskip(false),
         m_sensor(sensor),
         m_pvname(NULL),
         m_lvname(NULL),
         m_sensor_surface_index(0),
         m_parts(NULL),
         m_parallel_node(NULL)
{
}


OpticksCSG_t GVolume::getCSGFlag()
{
    return m_csgflag ; 
}

bool GVolume::isCSGSkip()
{
    return m_csgskip ; 
}
void GVolume::setCSGSkip(bool csgskip)
{
    m_csgskip = csgskip ; 
}


unsigned int GVolume::getBoundary() const 
{
    return m_boundary ; 
}

NSensor* GVolume::getSensor()
{
    return m_sensor ; 
}


void GVolume::setPVName(const char* pvname)
{
    m_pvname = strdup(pvname);
}
void GVolume::setLVName(const char* lvname)
{
    m_lvname = strdup(lvname);
}

const char* GVolume::getPVName()
{
    return m_pvname ; 
}
const char* GVolume::getLVName()
{
    return m_lvname ; 
}








void GVolume::Summary(const char* msg )
{
   if(!msg) msg = getDescription();
   if(!msg) msg = "GVolume::Summary" ;
   printf("%s\n", msg );
}


std::string GVolume::description()
{
    const char* desc_ = getDescription() ;

    std::string desc ;
    if(desc_) desc.assign(desc_);

  
    return desc; 
}


GParts* GVolume::getParts()
{
    return m_parts ;  
}
void GVolume::setParts(GParts* pts)
{
    m_parts = pts ; 
}


// ancillary slot for a parallel node tree, used by X4PhysicalVolume
void* GVolume::getParallelNode() const 
{
    return m_parallel_node ; 
}
void GVolume::setParallelNode(void* pnode)
{
    m_parallel_node = pnode ; 
}
 



void GVolume::setCSGFlag(OpticksCSG_t csgflag)
{
    m_csgflag = csgflag ; 
}

void GVolume::setBoundary(unsigned int boundary)
{
    m_boundary = boundary ; 
    setBoundaryIndices( boundary );
}


void GVolume::setBoundaryAll(unsigned boundary)
{
     unsigned nchild = getNumChildren();
     if(nchild > 0)
     {
        for(unsigned i=0 ; i < nchild ; i++)
        {
            GNode* node = getChild(i);
            GVolume* sub = dynamic_cast<GVolume*>(node);
            sub->setBoundary(boundary);
        }
     } 
}


void GVolume::setSensor(NSensor* sensor)
{
    m_sensor = sensor ; 
    // every triangle needs a value... use 0 to mean unset, so sensor   
    setSensorIndices( NSensor::RefIndex(sensor) );
}

guint4 GVolume::getIdentity()
{
    return guint4(
                   m_index, 
                   getMeshIndex(), 
                   m_boundary,
                   getSensorSurfaceIndex()
                 );
}


/*
void GVolume::setIdentity(const guint4& id )
{
    assert( id.x == m_index );
    assert( id.y == getMeshIndex() ) ;

    setBoundary( id.z );
    setSensorSurfaceIndex( id.w ); 
}
*/




void GVolume::setSensorSurfaceIndex(unsigned int ssi)
{
    m_sensor_surface_index = ssi ; 
}
unsigned int GVolume::getSensorSurfaceIndex()
{
    return m_sensor_surface_index ; 
}





 


void GVolume::Dump( const std::vector<GVolume*>& solids, const char* msg )
{
    unsigned numSolid = solids.size() ;
    LOG(info) << msg << " numSolid " << numSolid ; 
    for(unsigned i=0 ; i < numSolid ; i++) solids[i]->dump(); 
}


