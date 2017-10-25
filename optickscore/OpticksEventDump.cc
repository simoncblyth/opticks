
#include <iostream>
#include "RecordsNPY.hpp"
#include "NGLM.hpp"
#include "GLMFormat.hpp"

#include "Opticks.hh"
#include "OpticksEvent.hh"
#include "OpticksEventDump.hh"

#include "PLOG.hh"

OpticksEventDump::OpticksEventDump(OpticksEvent* evt ) 
    :
    m_ok(evt->getOpticks()),
    m_evt(evt),
    m_noload( evt ? evt->isNoLoad() : true ),
    m_records(NULL)
{
    init();
}


void OpticksEventDump::init()
{
    assert(m_ok);
    setupRecordsNPY();
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

    std::cout 
        << "TagDir:" << m_evt->getTagDir() << std::endl 
        << "ShapeString:" << m_evt->getShapeString() << std::endl 
        << "Loaded " << ( m_noload ? "NO" : "YES" ) 
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




void OpticksEventDump::setupRecordsNPY()
{
    if(m_noload || m_records) return ; 

    NPY<short>* rx = m_evt->getRecordData();
    assert(rx && rx->hasData());
    unsigned maxrec = m_evt->getMaxRec() ;

    Types* types = m_ok->getTypes();
    Typ* typ = m_ok->getTyp();

    RecordsNPY* rec = new RecordsNPY(rx, maxrec);

    rec->setTypes(types);
    rec->setTyp(typ);
    rec->setDomains(m_evt->getFDomain()) ;

    LOG(info) << "OpticksEvent::setupRecordsNPY " 
              << " shape " << rx->getShapeString() 
              ;

    m_evt->setRecordsNPY(rec);
    m_records = rec ; 
} 



