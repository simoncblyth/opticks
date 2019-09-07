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


#include <iostream>
#include "RecordsNPY.hpp"
#include "NGLM.hpp"
#include "GLMFormat.hpp"

#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksEventStat.hh"
#include "OpticksEventDump.hh"

#include "PLOG.hh"

OpticksEventDump::OpticksEventDump(OpticksEvent* evt ) 
    :
    m_ok(evt->getOpticks()),
    m_evt(evt),
    m_stat(new OpticksEventStat(evt,0)),
    m_noload( evt ? evt->isNoLoad() : true ),
    m_records( evt ? evt->getRecordsNPY() : NULL ),
    m_photons( evt ? evt->getPhotonData() : NULL ),
    m_seq( evt ? evt->getSequenceData() : NULL),
    m_num_photons(m_photons ? m_photons->getShape(0) : 0 )
{
    init();
}

void OpticksEventDump::init()
{
    assert(m_ok);

    if( m_photons && m_seq )
    {
        assert( m_photons->getShape(0)  == m_seq->getShape(0)  );
    }

}

unsigned OpticksEventDump::getNumPhotons() const 
{
    return m_num_photons ; 
}

void OpticksEventDump::Summary(const char* msg) const 
{
    LOG(info) << msg ; 
    const char* geopath = m_evt->getGeoPath();
    std::cout << m_photons->description() << std::endl ;

    std::cout 
        << std::setw(20) 
        << "TagDir:" 
        << m_evt->getTagDir() 
        << std::endl 
        << std::setw(20) 
        << "ShapeString:" << m_evt->getShapeString() 
        << std::endl 
        << std::setw(20) 
        << "Loaded " << ( m_noload ? "NO" : "YES" )   
        << std::endl
        << std::setw(20) 
        << "GeoPath " 
        << ( geopath ? geopath : "-" ) 
        << std::endl 
        ;


    if(m_noload) return ; 

    LOG(info) << "evt->Summary()" ; 
    m_evt->Summary() ; 
}



void OpticksEventDump::dump(unsigned photon_id) const 
{
    if(m_noload) return ; 
    LOG(info) 
         << " tagdir " << m_evt->getTagDir()
         << " photon_id " << photon_id ; 
    dumpRecords(photon_id);
    dumpPhotonData(photon_id);
}

void OpticksEventDump::dumpRecords(unsigned photon_id ) const 
{
    if( photon_id >= m_num_photons ) return ; 

    std::vector<NRec> recs ; 
    glm::vec4 ldd = m_records->getLengthDistanceDurationRecs(recs, photon_id ); 
    assert( ldd.x >= 0.f );

    for(unsigned p=0 ; p < recs.size() ; p++)
    {
        const NRec& rec = recs[p] ; 
        if( rec.post.w == 0 ) continue ; 


        //unsigned hflg = rec.flag.w ;  //    3: (MISS = 0x1 << 2)   13: (TORCH = 0x1 << 12)

        std::cout 
            << std::setw(40) 
            << gpresent(rec.post,2,11) 
            << std::setw(40) 
            << gpresent(rec.polw,2,7) 
            << std::setw(40) 
            << gpresent( rec.flag )  // m1, m2, bnd, hflg     
            << std::setw(10) 
            << rec.hs
            << std::setw(10) 
            << rec.m1
            << std::setw(10) 
            << rec.m2
            << std::endl 
            ;
    }
}


void OpticksEventDump::dumpPhotonData(unsigned photon_id) const 
{
    if( photon_id >= m_num_photons ) return ; 
    
    unsigned i = photon_id ; 
    unsigned int ux = m_photons->getUInt(i,0,0); 
    float fx = m_photons->getFloat(i,0,0); 
    float fy = m_photons->getFloat(i,0,1); 
    float fz = m_photons->getFloat(i,0,2); 
    float fw = m_photons->getFloat(i,0,3); 

    printf(" ph  %7u   ux %7u   fxyzw %10.3f %10.3f %10.3f %10.3f \n", i, ux, fx, fy, fz, fw );             
}



