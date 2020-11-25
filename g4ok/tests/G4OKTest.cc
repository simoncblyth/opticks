#include <cassert>
#include "OPTICKS_LOG.hh"

#include "G4PVPlacement.hh"
#include "G4Opticks.hh"
#include "G4OpticksHit.hh"

#include "MockSensorAngularEfficiencyTable.hh"

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

    ##G4OKTest --torchtarget 3153
    G4OKTest 

Following implementation of passing GDML auxiliary info 
thru the geocache now get the default torch target from the GDML 
rather than having to specify it on the commandline.  

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

    collectTorchStep(unsigned num_photons, int node_index);  


**/


class G4OKTest 
{
        static const plog::Severity LEVEL ; 
    public:
        G4OKTest(int argc, char** argv);
    private:
        int  initLog(int argc, char** argv);
        void init();
        void initGeometry();
        void initSensorData();
        void initSensorAngularEfficiency();
        void saveSensorLib() const ;
    public: 
        unsigned getNumGenstepPhotons(int eventID) const ;
        void     collectGensteps(int eventID); 
        void     propagate(int eventID); 
        void     checkHits(int eventID) const ; 
        int      rc() const ; 
    private:
        int          m_log ; 
        const char*  m_gdmlpath ; 
        int          m_torchtarget ; 
        G4Opticks*   m_g4ok ; 
        bool         m_debug ; 
        const char*  m_tmpdir ; 
};



const plog::Severity G4OKTest::LEVEL = PLOG::EnvLevel("G4OKTest", "DEBUG"); 

G4OKTest::G4OKTest(int argc, char** argv)
    :
    m_log(initLog(argc, argv)),
    m_gdmlpath(PLOG::instance->get_arg_after("--gdmlpath", NULL)),
    m_torchtarget(PLOG::instance->get_int_after("--torchtarget", "-1")),
    m_g4ok(new G4Opticks),
    m_debug(true),
    m_tmpdir("$TMP/G4OKTest")
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
    initGeometry();
    initSensorData();
    initSensorAngularEfficiency();
    if(m_debug) saveSensorLib(); 
    m_g4ok->uploadSensorLib(); 

    // m_g4ok->snap(m_tmpdir);   // snapping before event upload fails due to invalid context : generate.cu requires sequence_buffer

}


/**
G4OKTest::initGeometry
-----------------------

When no gdmlpath argument is provided the geometry is 
loaded from the geocache identified by OPTICKS_KEY envvar.

**/

void G4OKTest::initGeometry()
{
    if(m_gdmlpath == NULL)
    {
        m_g4ok->loadGeometry(); 
    }
    else
    {
        m_g4ok->setGeometry(m_gdmlpath);  
    }
}

/**
G4OKTest::initSensorData
--------------------------

When loading geometry from cache there is no Geant4 tree of volumes in memory resulting 
in the vector of sensor placements being empty. However the number of sensors is available
allowing a standin sensor identifier to be assigned to each sensor. Currently using the
Opticks triplet identifier for this.

**/

void G4OKTest::initSensorData()
{
    bool loaded = m_g4ok->isLoadedFromCache(); 
    const std::vector<G4PVPlacement*>& sensor_placements = m_g4ok->getSensorPlacements() ;
    unsigned num_sensor = m_g4ok->getNumSensorVolumes(); 
    assert( sensor_placements.size() == ( loaded ? 0 : num_sensor )); 

    unsigned num_distinct_copynumber = m_g4ok->getNumDistinctPlacementCopyNo();  
    bool use_standin = num_distinct_copynumber == 1 ; 

    LOG(LEVEL)
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
        int   sensor_cat = 0 ;        // must be less than num_cat, -1 when no angular efficiency info

        if(dump) std::cout 
            << " sensor_index(dec) "      << std::setw(5) << std::dec << sensor_index
            << " (hex) "                  << std::setw(5) << std::hex << sensor_index << std::dec
            << " sensor_identifier(hex) " << std::setw(7) << std::hex << sensor_identifier << std::dec
            << " standin(hex) "           << std::setw(7) << std::hex << sensorIdentityStandin << std::dec
            << std::endl
            ;

        m_g4ok->setSensorData( sensor_index, efficiency_1, efficiency_2, sensor_cat, sensor_identifier );
    }

    LOG(LEVEL) << "] setSensorData num_sensor " << num_sensor ; 
}

void G4OKTest::initSensorAngularEfficiency()
{
    unsigned num_sensor_cat = 1 ; 
    unsigned num_theta_steps = 180 ;  // height
    unsigned num_phi_steps = 360 ;    // width 

    NPY<float>* tab = MockSensorAngularEfficiencyTable::Make(num_sensor_cat, num_theta_steps, num_phi_steps); 
    m_g4ok->setSensorAngularEfficiency( tab ); 
}

void G4OKTest::saveSensorLib() const
{
    const char* dir = "$TMP/G4OKTest/SensorLib" ; 
    m_g4ok->saveSensorLib(dir); 
    LOG(info) << dir ; 
}









unsigned G4OKTest::getNumGenstepPhotons(int eventID) const
{
    unsigned num_photons = 0 ;   // 0: leads to default torch genstep num_photons of 10000(?)
    switch(eventID) 
    {
       case 0:  num_photons = 5000 ; break ; 
       case 1:  num_photons = 2000 ; break ; 
       case 2:  num_photons = 3000 ; break ; 
       case 3:  num_photons = 4000 ; break ; 
       case 4:  num_photons = 5000 ; break ; 
       case 5:  num_photons = 5000 ; break ; 
       case 6:  num_photons = 4000 ; break ; 
       case 7:  num_photons = 3000 ; break ; 
       case 8:  num_photons = 2000 ; break ; 
       case 9:  num_photons = 1000 ; break ;
       default: num_photons = 0    ; break ;  
    }
    return num_photons ; 
}


void G4OKTest::collectGensteps(int eventID)
{
    unsigned num_genstep_photons = getNumGenstepPhotons(eventID); 
    int node_index = m_torchtarget ; 

    m_g4ok->collectDefaultTorchStep(num_genstep_photons, node_index); 

    LOG(error) 
        << " eventID " << eventID
        << " num_genstep_photons " << num_genstep_photons
        ;  
}

void G4OKTest::propagate(int eventID)
{
    LOG(LEVEL) << "[" ; 
    int num_hit = m_g4ok->propagateOpticalPhotons(eventID);
    unsigned num_genstep_photons = getNumGenstepPhotons(eventID); 

    LOG(error) 
        << " eventID " << eventID
        << " num_genstep_photons " << num_genstep_photons
        << " num_hit " << num_hit 
        ; 

    m_g4ok->dumpHits("G4OKTest::propagate"); 

    checkHits(eventID); 

    m_g4ok->reset(); // <--- without reset gensteps just keep accumulating 



    if( eventID == 0 )
    {       
        m_g4ok->snap(m_tmpdir, "snap");
    }
    LOG(LEVEL) << "]" ; 
}

void G4OKTest::checkHits(int eventID) const 
{
    G4OpticksHit hit ; 
    unsigned num_hit = m_g4ok->getNumHit(); 
    LOG(info) 
        << " eventID " << eventID
        << " num_hit " << num_hit
        ; 


    for(unsigned i=0 ; i < num_hit ; i++)
    {
        m_g4ok->getHit(i, &hit); 
        std::cout 
            << std::setw(5) << i 
            << " hit.boundary "           << std::setw(4) << hit.boundary 
            << " hit.sensor_index "       << std::setw(5) << hit.sensor_index 
            << " hit.node_index "         << std::setw(5) << hit.node_index 
            << " hit.photon_index "       << std::setw(5) << hit.photon_index 
            << " hit.flag_mask    "       << std::setw(10) << std::hex << hit.flag_mask  << std::dec
            << " hit.sensor_identifier "  << std::setw(10) << std::hex << hit.sensor_identifier << std::dec
            << " hit.weight "             << std::setw(5) << hit.weight 
            << " hit.wavelength "         << std::setw(8) << hit.wavelength 
            << " hit.time "               << std::setw(8) << hit.time
         //   << " hit.local_position " << hit.local_position 
            << std::endl 
            ;    
        // G4ThreeVector formatter doesnt play well with setw
    }
}


int G4OKTest::rc() const 
{
    return 0 ; 
}



int main(int argc, char** argv)
{
    G4OKTest t(argc, argv); 

    for(int ievt=0 ; ievt < 1 ; ievt++)
    {
       t.collectGensteps(ievt);
       t.propagate(ievt);
    }

    return t.rc() ;
}


