#include "OpticksEvent.hh"

// npy-
#include "NPY.hpp"
#include "GLMPrint.hpp"
#include "RecordsNPY.hpp"
#include "BLog.hh"

#include <cassert>
#include <iostream>


void test_genstep_derivative()
{
    OpticksEvent evt("cerenkov", "1", "dayabay", "") ;

    NPY<float>* trk = evt.loadGenstepDerivativeFromFile("track");
    assert(trk);

    LOG(info) << trk->getShapeString();

    glm::vec4 origin    = trk->getQuad(0,0) ;
    glm::vec4 direction = trk->getQuad(0,1) ;
    glm::vec4 range     = trk->getQuad(0,2) ;

    print(origin,"origin");
    print(direction,"direction");
    print(range,"range");


}


void test_genstep()
{   
    OpticksEvent evt("cerenkov", "1", "dayabay", "") ;

    evt.setGenstepData(evt.loadGenstepFromFile());

    evt.dumpPhotonData();

}

void test_load_and_records()
{  
    const char* typ = "torch" ; 
    const char* tag = "4" ; 
    const char* det = "dayabay" ; 
    const char* cat = "PmtInBox" ; 
 
    OpticksEvent* m_evt = new OpticksEvent(typ, tag, det, cat) ;
    m_evt->loadBuffers();


    unsigned int maxrec = m_evt->getMaxRec() ;

    NPY<short>* rx = m_evt->getRecordData();
    assert(rx && rx->hasData());

    std::cout << "test_load rx " 
              << " shape " << rx->getShapeString() 
              << std::endl ;  
    

    RecordsNPY* m_rec ; 
    
    bool flat = false ; 
    m_rec = new RecordsNPY(rx, m_evt->getMaxRec(), flat);
    //m_rec->setTypes(types);
    //m_rec->setTyp(typ);
    m_rec->setDomains(m_evt->getFDomain()) ;

    for(unsigned int photon_id=0 ; photon_id < 10 ; photon_id++ )
    {
        for(unsigned int r=0 ; r < maxrec ; r++)
        {
            //unsigned int record_id = photon_id*m_maxrec + r ;
            unsigned int i = photon_id ;
            unsigned int j = r ;

            m_rec->dumpRecord(i,j,"dumpRecord (i,j)");
        }
    }
}

void test_load()
{
}




int main(int argc, char** argv)
{
    NLog nl("OpticksEventTest.log","info");
    nl.configure(argc, argv, "/tmp"); 

    //test_genstep_derivative();
    //test_genstep();
    //test_load_and_records();
    test_load();
    return 0 ;
}
