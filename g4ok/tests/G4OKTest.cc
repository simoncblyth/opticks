#include <cassert>
#include "OPTICKS_LOG.hh"

#include "G4PVPlacement.hh"
#include "G4Opticks.hh"

/**
G4OKTest
===============

This is aiming to replace::

   okg4/tests/OKX4Test.cc 

in order to reduce duplicated code between G4Opticks and here
and make G4Opticks functionality testable without ascending to the 
detector specific level.

Notice that this is not using the cache::

    opticksaux-;G4OKTest --gdmlpath $(opticksaux-dx0) --x4polyskip 211,232

    G4OPTICKS_DEBUG="--x4polyskip 211,232" lldb_ G4OKTest --  --gdmlpath $(opticksaux-dx0) 
    ## testing embedded, so options need to be passed in the back door 
    ## no sensors found with dx0, so try jv5
 
    G4OPTICKS_DEBUG="" lldb_ G4OKTest --  --gdmlpath $(opticksaux-jv5) 


Hmm not using the cache makes this not very convenient for dev cycling

**/

struct G4OKTest 
{
    G4OKTest(int argc, char** argv);
    int  initLog(int argc, char** argv);
    void init();
    int rc() const ; 

    int          m_log ; 
    const char*  m_gdmlpath ; 
    G4Opticks*   m_g4opticks ; 
};


int G4OKTest::initLog(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    return 0 ; 
}

G4OKTest::G4OKTest(int argc, char** argv)
    :
    m_log(initLog(argc, argv)),
    m_gdmlpath(PLOG::instance->get_arg_after("--gdmlpath", NULL)),
    m_g4opticks(new G4Opticks)
{
    init();
}


/**
G4OKTest::init
----------------

This is usually done from detector specific code such as:: 

    Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstruction.cc

**/

void G4OKTest::init()
{
    if( m_gdmlpath == NULL ) return ; 
    LOG(info) << m_gdmlpath ;  
    G4Opticks* g4opticks = m_g4opticks ; 

    g4opticks->setGeometry(m_gdmlpath);  
    const std::vector<G4PVPlacement*>& sensor_placements = g4opticks->getSensorPlacements() ;
    unsigned num_sensor = sensor_placements.size();
    LOG(info) << "[ setSensorData num_sensor " << num_sensor ; 

    for(unsigned i=0 ; i < num_sensor ; i++)
    {
        unsigned sensor_index = i ;
        const G4PVPlacement* pv = sensor_placements[sensor_index] ;
        G4int copyNo = pv->GetCopyNo();

        float efficiency_1 = 0.5f ;    
        float efficiency_2 = 1.0f ;    
        int   sensor_cat = 0 ; 
        int   sensor_identifier = copyNo ;

        std::cout 
            << " sensor_index " << sensor_index
            << " sensor_identifier " << sensor_identifier
            << std::endl
            ;

        g4opticks->setSensorData( sensor_index, efficiency_1, efficiency_2, sensor_cat, sensor_identifier );
    }
    std::string SensorCategoryList = "placeholder" ; 
    g4opticks->setSensorDataMeta<std::string>("SensorCategoryList", SensorCategoryList);
    
    LOG(info) << "] setSensorData num_sensor " << num_sensor ; 

    //g4opticks->setSensorAngularEfficiency...

    LOG(info) << g4opticks->dbgdesc() ; 
}

int G4OKTest::rc() const 
{
    return 0 ; 
}


int main(int argc, char** argv)
{
    G4OKTest g4okt(argc, argv); 
    return g4okt.rc() ;
}


