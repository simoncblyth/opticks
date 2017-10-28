
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
    m_records( evt ? evt->getRecordsNPY() : NULL )
{
    init();
}

void OpticksEventDump::init()
{
    assert(m_ok);
}

void OpticksEventDump::dump(const char* msg)
{
    Summary(msg);
    dumpRecords();
    dumpPhotonData();
}

void OpticksEventDump::Summary(const char* msg)
{
    LOG(info) << msg ; 
    const char* geopath = m_evt->getGeoPath();

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



void OpticksEventDump::dumpRecords(const char* msg)
{
    for(unsigned photon_id=0 ; photon_id < 5 ; photon_id++ ) dumpRecords(msg, photon_id );
}

void OpticksEventDump::dumpRecords(const char* msg, unsigned photon_id )
{
    LOG(info) << msg ; 
    if(m_noload) return ; 

    unsigned maxrec = m_evt->getMaxRec() ;
    for(unsigned r=0 ; r < maxrec ; r++)
    {
        m_records->dumpRecord(photon_id,r,"dumpRecord (i,j)");
    }

    std::vector<glm::vec4> posts ; 
    glm::vec4 ldd = m_records->getLengthDistanceDurationPosts(posts, photon_id ); 

    for(unsigned p=0 ; p < posts.size() ; p++)
        std::cout << gpresent( "post", posts[p] ) ; 
}


void OpticksEventDump::dumpPhotonData(const char* msg)
{
    LOG(info) << msg ; 
    if(m_noload) return ; 

    NPY<float>* photons = m_evt->getPhotonData();
    if(!photons) return ;
    dumpPhotonData(photons);
}

void OpticksEventDump::dumpPhotonData(NPY<float>* photons)
{
    std::cout << photons->description("OpticksEventDump::dumpPhotonData") << std::endl ;

    for(unsigned int i=0 ; i < photons->getShape(0) ; i++)
    {
        //if(i%10000 == 0)
        if(i < 10)
        {
            unsigned int ux = photons->getUInt(i,0,0); 
            float fx = photons->getFloat(i,0,0); 
            float fy = photons->getFloat(i,0,1); 
            float fz = photons->getFloat(i,0,2); 
            float fw = photons->getFloat(i,0,3); 
            printf(" ph  %7u   ux %7u   fxyzw %10.3f %10.3f %10.3f %10.3f \n", i, ux, fx, fy, fz, fw );             
        }
    }  
}



