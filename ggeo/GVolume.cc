/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

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


GVolume::GVolume( unsigned index, GMatrix<float>* transform, const GMesh* mesh )
    : 
    GNode(index, transform, mesh ),
    m_boundary(-1),
    m_csgflag(CSG_PARTLIST),
    m_csgskip(false),
    m_sensor(NULL),
    m_pvname(NULL),
    m_lvname(NULL),
    m_sensor_surface_index(0),
    m_parts(NULL),
    m_pt(NULL),
    m_parallel_node(NULL), 
    m_copyNumber(-1)
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


unsigned GVolume::getBoundary() const 
{
    return m_boundary ; 
}

NSensor* GVolume::getSensor()
{
    return m_sensor ; 
}

void GVolume::setCopyNumber(unsigned copyNumber)
{
    m_copyNumber = copyNumber ; 
}
unsigned GVolume::getCopyNumber() const 
{
    return m_copyNumber ; 
}

void GVolume::setPVName(const char* pvname)
{
    m_pvname = strdup(pvname);
}
void GVolume::setLVName(const char* lvname)
{
    m_lvname = strdup(lvname);
}

const char* GVolume::getPVName() const 
{
    return m_pvname ; 
}
const char* GVolume::getLVName() const 
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


GPt* GVolume::getPt() const 
{
    return m_pt ;  
}
void GVolume::setPt(GPt* pt)
{
    m_pt = pt ; 
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
    unsigned node_index = m_index ;    
  
    //unsigned identity_index = getSensorSurfaceIndex() ;   
    unsigned identity_index = m_copyNumber  ;   

    return guint4(
                   node_index, 
                   getMeshIndex(), 
                   m_boundary,
                   identity_index
                 );
}


//void GVolume::setIdentity(const guint4& id )
//{
//    assert( id.x == m_index );
//    assert( id.y == getMeshIndex() ) ;
//
//    setBoundary( id.z );
//    setSensorSurfaceIndex( id.w ); 
//}

void GVolume::setSensorSurfaceIndex(unsigned int ssi)
{
    m_sensor_surface_index = ssi ; 
}
unsigned int GVolume::getSensorSurfaceIndex()
{
    return m_sensor_surface_index ; 
}

void GVolume::Dump( const std::vector<GVolume*>& volumes, const char* msg )
{
    unsigned numVolume = volumes.size() ;
    LOG(info) << msg << " numVolume " << numVolume ; 
    for(unsigned i=0 ; i < numVolume ; i++) volumes[i]->dump(); 
}

