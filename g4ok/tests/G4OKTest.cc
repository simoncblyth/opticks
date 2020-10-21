#include <cassert>
#include "OPTICKS_LOG.hh"

#include "G4PVPlacement.hh"
#include "G4Opticks.hh"

/**
G4OKTest
===============

This test is intended to provide a way of testing G4Opticks machinery 
without ascending to the level of the experiment.

Geometry Setup
---------------

This can now run either from GDML file OR from geocache.
When a --gdmlpath argument is used this will parse the GDML 
into a Geant4 geometry and translate that into Opticks GGeo.   
Without the --gdmlpath argument the cache identified by 
OPTICKS_KEY envvar will be loaded.
 
From GDML::

    opticksaux-
    G4OKTest --gdmlpath $(opticksaux-dx1)
    G4OPTICKS_DEBUG="--x4polyskip 211,232" lldb_ G4OKTest --  --gdmlpath $(opticksaux-dx1) 
    G4OPTICKS_DEBUG="" lldb_ G4OKTest --  --gdmlpath $(opticksaux-jv5) 

From cache(experimental)::

    G4OKTest --torchtarget 3153

Notice that Opticks is "embedded" when running from GDML which means that 
it does not parse the commandline, this matches production usage. In order  
supply Opticks additional arguments the G4OPTICKS_DEBUG envvar backdoor can be used.

When running from cache the commandline is now parsed using the 
arguments captured by OPTICKS_LOG. Caution that both arguments
from the commandline and the fixed embedded commandline are used. 
This prevents the use of some arguments as duplicates cause asserts.  


Genstep Setup
---------------

Hmm: Opticks can generate torch gensteps out of nowhere, but the 
point of G4OKTest is to test G4Opticks : so need to do things 
more closely to the production workflow. 

So add::

    collectTorchStep(unsigned target_nidx, unsigned num_photons);  




**/

class G4OKTest 
{
    public:
        G4OKTest(int argc, char** argv);
    private:
        int  initLog(int argc, char** argv);
        void init();
    public: 
        void collect(); 
        void propagate(); 
        int rc() const ; 
    private:
        int          m_log ; 
        const char*  m_gdmlpath ; 
        int          m_torchtarget ; 
        G4Opticks*   m_g4ok ; 
};


G4OKTest::G4OKTest(int argc, char** argv)
    :
    m_log(initLog(argc, argv)),
    m_gdmlpath(PLOG::instance->get_arg_after("--gdmlpath", NULL)),
    m_torchtarget(PLOG::instance->get_int_after("--torchtarget", "0")),
    m_g4ok(new G4Opticks)
{
    init();
}

int G4OKTest::initLog(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    return 0 ; 
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

    unsigned num_distinct_copynumber = m_g4ok->getNumDistinctPlacementCopyNo();  
    bool use_standin = num_distinct_copynumber == 1 ; 

    LOG(info)
        << "[ setSensorData num_sensor " << num_sensor 
        << " num_distinct_copynumber " << num_distinct_copynumber
        << " Geometry " << ( loaded ? "LOADED FROM CACHE" : "LIVE TRANSLATED" )
        ;

    bool dump = false ; 
    for(unsigned i=0 ; i < num_sensor ; i++)
    {
        unsigned sensor_index = i ;
        unsigned sensorIdentityStandin = m_g4ok->getSensorIdentityStandin(sensor_index); // opticks triplet identifier

        const G4PVPlacement* pv = loaded ? NULL : sensor_placements[sensor_index] ;
        int   sensor_identifier = ( use_standin || pv == NULL) ? sensorIdentityStandin : pv->GetCopyNo() ;

        // GDML physvol/@copynumber attribute persists the CopyNo : but this defaults to 0 unless set at detector level

        float efficiency_1 = 0.5f ;    
        float efficiency_2 = 1.0f ;    
        int   sensor_cat = 0 ; 

        if(dump) std::cout 
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


void G4OKTest::collect()
{
    unsigned node_index = m_torchtarget ; 
    m_g4ok->collectDefaultTorchStep(node_index); 
}


void G4OKTest::propagate()
{
    G4int eventID = 0 ; 
    int nhit = m_g4ok->propagateOpticalPhotons(eventID);
    LOG(info) << " nhit " << nhit ; 
}

int G4OKTest::rc() const 
{
    return 0 ; 
}




int main(int argc, char** argv)
{
    G4OKTest t(argc, argv); 
    t.collect();
    t.propagate();
    return t.rc() ;
}


