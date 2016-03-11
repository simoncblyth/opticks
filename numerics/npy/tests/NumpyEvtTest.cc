#include "NumpyEvt.hpp"
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "RecordsNPY.hpp"
#include "NLog.hpp"

#include <cassert>
#include <iostream>

void test_genstep()
{   
    NumpyEvt evt("cerenkov", "1", "dayabay") ;

    evt.setGenstepData(evt.loadGenstepFromFile());

    evt.dumpPhotonData();
}

void test_load()
{  
    const char* typ = "torch" ; 
    const char* tag = "4" ; 
    const char* det = "dayabay" ; 
    const char* cat = "PmtInBox" ; 
 
    NumpyEvt* m_evt = new NumpyEvt(typ, tag, det, cat) ;
    m_evt->load();

    bool m_flat = m_evt->isFlat() ;
    assert(m_flat) ;

    unsigned int m_maxrec = m_evt->getMaxRec() ;

    NPY<short>* rx = m_evt->getRecordData();
    assert(rx && rx->hasData());

    std::cout << "test_load rx " 
              << " shape " << rx->getShapeString() 
              << " flat " << m_flat 
              << std::endl ;  
    

    RecordsNPY* m_rec ; 
    
    m_rec = new RecordsNPY(rx, m_evt->getMaxRec(), m_evt->isFlat());
    //m_rec->setTypes(types);
    //m_rec->setTyp(typ);
    m_rec->setDomains(m_evt->getFDomain()) ;

    for(unsigned int photon_id=0 ; photon_id < 10 ; photon_id++ )
    {
        for(unsigned int r=0 ; r < 10 ; r++)
        {
            unsigned int record_id = photon_id*m_maxrec + r ;
            unsigned int i = m_flat ? record_id : photon_id ;
            unsigned int j = m_flat ? 0         : r ;

            m_rec->dumpRecord(i,j,"dumpRecord (i,j)");
        }
    }


}


int main(int argc, char** argv)
{
    NLog nl("NumpyEvtTest.log","info");
    nl.configure(argc, argv, "/tmp"); 

    //test_genstep();
    test_load();
    return 0 ;
}
