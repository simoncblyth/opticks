
#include <iostream>
#include "RecordsNPY.hpp"

#include "OpticksEvent.hh"
#include "OpticksEventDump.hh"

#include "PLOG.hh"

OpticksEventDump::OpticksEventDump(OpticksEvent* evt ) 
   :
   m_evt(evt),
   m_noload( evt ? evt->isNoLoad() : true )
{
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
    LOG(info) << msg ; 
    if(m_noload) return ; 

    unsigned int maxrec = m_evt->getMaxRec() ;

    NPY<short>* rx = m_evt->getRecordData();
    assert(rx && rx->hasData());

    LOG(info) << "OpticksEventDump::dumpRecords " 
              << " shape " << rx->getShapeString() 
              ;

    RecordsNPY* rec ; 
    
    rec = new RecordsNPY(rx, maxrec);
    //m_rec->setTypes(types);
    //m_rec->setTyp(typ);
    rec->setDomains(m_evt->getFDomain()) ;

    for(unsigned photon_id=0 ; photon_id < 10 ; photon_id++ )
    {
        for(unsigned r=0 ; r < maxrec ; r++)
        {
            //unsigned int record_id = photon_id*m_maxrec + r ;
            unsigned i = photon_id ;
            unsigned j = r ;

            rec->dumpRecord(i,j,"dumpRecord (i,j)");
        }
    }

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




