#include <cassert>
#include "OPTICKS_LOG.hh"

#include "G4PVPlacement.hh"
#include "G4Opticks.hh"

/**
G4OKTest
===============

This test is intended to provide a way of testing G4Opticks machinery 
without ascending to the level of the experiment.

Formerly this could not run from geocache, it now can do 
both. When a --gdmlpath argument is used this will 
parse the GDML into a Geant4 geometry and translate that 
into Opticks GGeo.   Without the --gdmlpath argument the 
cache identified by OPTICKS_KEY envvar will be loaded.
 
From GDML::

    opticksaux-
    G4OKTest --gdmlpath $(opticksaux-dx1)
    G4OPTICKS_DEBUG="--x4polyskip 211,232" lldb_ G4OKTest --  --gdmlpath $(opticksaux-dx1) 
    G4OPTICKS_DEBUG="" lldb_ G4OKTest --  --gdmlpath $(opticksaux-jv5) 


From cache(experimental)::

    G4OKTest 

Notice that Opticks is "embedded" in both cases, thus 
need to supply Opticks arguments via G4OPTICKS_DEBUG envvar backdoor.

Hmm embedded running matches production usage, BUT its a pain, better for G4Opticks
to be passed Opticks ptr ? Which when NULL results in embedded running.

**/

struct G4OKTest 
{
    G4OKTest(int argc, char** argv);
    int  initLog(int argc, char** argv);
    void init();
    int rc() const ; 

    int          m_log ; 
    const char*  m_gdmlpath ; 
    G4Opticks*   m_g4ok ; 
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
    m_g4ok(new G4Opticks)
{
    init();
}


/**
G4OKTest::init
----------------

Code similar to this is usually within detector simulation frameworks, eg for JUNO:: 

    Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstruction.cc

The origin G4PVPlacement for sensor volumes are provided by Opticks in order 
that the corresponding detector specific sensor identifiers can be 
communicated to Opticks via setSensorData. For JUNO the sensor identifiers
can be simply obtained with G4PVPlacement::GetCopyNo.

When running from cache G4PVPlacement are not available, so the 
opticksTripletIdentifier is used as a standin for the sensor identifier. 

**/

void G4OKTest::init()
{
    if(m_gdmlpath == NULL)
    {
        m_g4ok->loadGeometry(); 
    }
    else
    {
        m_g4ok->setGeometry(m_gdmlpath);  
    }

    bool loaded = m_g4ok->isLoadedFromCache(); 
    const std::vector<G4PVPlacement*>& sensor_placements = m_g4ok->getSensorPlacements() ;
    unsigned num_sensor = m_g4ok->getNumSensorVolumes(); 
    assert( sensor_placements.size() == ( loaded ? 0 : num_sensor )); 

    LOG(info)
        << "[ setSensorData num_sensor " << num_sensor 
        << " Geometry " << ( loaded ? "LOADED FROM CACHE" : "LIVE TRANSLATED" )
        ;

    for(unsigned i=0 ; i < num_sensor ; i++)
    {
        unsigned sensor_index = i ;
        unsigned sensorIdentityStandin = m_g4ok->getSensorIdentityStandin(sensor_index); // opticks triplet identifier

        const G4PVPlacement* pv = loaded ? NULL                  : sensor_placements[sensor_index] ;
        int   sensor_identifier = loaded ? sensorIdentityStandin : pv->GetCopyNo()                 ;

        float efficiency_1 = 0.5f ;    
        float efficiency_2 = 1.0f ;    
        int   sensor_cat = 0 ; 

        std::cout 
            << " sensor_index(dec) "      << std::setw(5) << std::dec << sensor_index
            << " (hex) "                  << std::setw(5) << std::hex << sensor_index << std::dec
            << " sensor_identifier(hex) " << std::setw(7) << std::hex << sensor_identifier << std::dec
            << " standin(hex) "           << std::setw(7) << std::hex << sensorIdentityStandin << std::dec
            << std::endl
            ;

        m_g4ok->setSensorData( sensor_index, efficiency_1, efficiency_2, sensor_cat, sensor_identifier );
    }
    std::string SensorCategoryList = "placeholder" ; 
    m_g4ok->setSensorDataMeta<std::string>("SensorCategoryList", SensorCategoryList);
    
    LOG(info) << "] setSensorData num_sensor " << num_sensor ; 

    //m_g4ok->setSensorAngularEfficiency...

    LOG(info) << m_g4ok->dbgdesc() ; 
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


