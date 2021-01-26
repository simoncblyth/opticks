#include <cassert>
#include "OPTICKS_LOG.hh"

#include "G4PVPlacement.hh"
#include "G4Opticks.hh"
#include "G4OpticksHit.hh"
#include "OpticksFlags.hh"

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
    OPTICKS_EMBEDDED_COMMANDLINE_EXTRA="--x4polyskip 211,232" lldb_ G4OKTest --  --gdmlpath $(opticksaux-dx1) 
    OPTICKS_EMBEDDED_COMMANDLINE_EXTRA="" lldb_ G4OKTest --  --gdmlpath $(opticksaux-jv5) 

From cache(experimental)::

    ##G4OKTest --torchtarget 3153
    G4OKTest 

Following implementation of passing GDML auxiliary info 
thru the geocache now get the default torch target from the GDML 
rather than having to specify it on the commandline.  

Notice that Opticks is "embedded" when running from GDML which means that 
it does not parse the commandline, this matches production usage. In order  
to control the Opticks commandline the below envvars can be used.

OPTICKS_EMBEDDED_COMMANDLINE
   default when not defined is "pro" 

OPTICKS_EMBEDDED_COMMANDLINE_EXTRA
   adds to the commandline

Note that duplicate arguments cause asserts.  


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
        void initCommandLine();
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
        bool         m_snap ; 
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
    m_snap(PLOG::instance->has_arg("--snap")),
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
    initCommandLine(); 
    initGeometry();
    initSensorData();
    initSensorAngularEfficiency();
    if(m_debug) saveSensorLib(); 
    //m_g4ok->snap(m_tmpdir);   // snapping before event upload fails due to invalid context : generate.cu requires sequence_buffer
}


/**
G4OKTest::initCommandLine
---------------------------

HMM how to handle detector specific options like --pvname ... --boundary ... needed for WAY_BUFFER ?
Actually in normal G4Opticks usage this is a non-problem as a single detector geometry is implicitly assumed, 
so can simply use detector specific commandline options with setEmbeddedCommandLineExtra.

G4OKTest is unusual in that it attempts to work for multiple geometries. 
One way is to somehow include these settings within geocache (or GDMLAux) metadata, 
so each geometry auto-gets the needed config without having to painfully discern the
detector then pick between hardcoded options.

**/

void G4OKTest::initCommandLine()
{
    const char* extra = NULL ; 
    m_g4ok->setEmbeddedCommandLineExtra(extra); 
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
        unsigned sensorIndex = 1+i ;  // 1-based
        unsigned sensorIdentityStandin = m_g4ok->getSensorIdentityStandin(sensorIndex); // opticks triplet identifier

        const G4PVPlacement* pv = loaded ? NULL : sensor_placements[sensorIndex-1] ;
        int   sensor_identifier = ( use_standin || pv == NULL) ? sensorIdentityStandin : pv->GetCopyNo() ;

        // GDML physvol/@copynumber attribute persists the CopyNo : but this defaults to 0 unless set at detector level

        float efficiency_1 = 0.5f ;    
        float efficiency_2 = 1.0f ;    
        int   sensor_cat = 0 ;        // must be less than num_cat, -1 when no angular efficiency info

        if(dump) std::cout 
            << " sensorIndex(dec) "       << std::setw(5) << std::dec << sensorIndex
            << " (hex) "                  << std::setw(5) << std::hex << sensorIndex << std::dec
            << " sensor_identifier(hex) " << std::setw(7) << std::hex << sensor_identifier << std::dec
            << " standin(hex) "           << std::setw(7) << std::hex << sensorIdentityStandin << std::dec
            << std::endl
            ;

        m_g4ok->setSensorData( sensorIndex, efficiency_1, efficiency_2, sensor_cat, sensor_identifier );
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
    m_g4ok->saveSensorLib(m_tmpdir, "SensorLib"); 
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

    if( m_snap && eventID == 0 )
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
            << " boundary "           << std::setw(4) << hit.boundary 
            << " sensorIndex "        << std::setw(5) << hit.sensorIndex 
            << " nodeIndex "          << std::setw(5) << hit.nodeIndex 
            << " photonIndex "        << std::setw(5) << hit.photonIndex 
            << " flag_mask    "       << std::setw(10) << std::hex << hit.flag_mask  << std::dec
            << " sensor_identifier "  << std::setw(10) << std::hex << hit.sensor_identifier << std::dec
         // << " weight "             << std::setw(5) << hit.weight 
            << " wavelength "         << std::setw(8) << hit.wavelength 
            << " time "               << std::setw(8) << hit.time
         // << " local_position " << hit.local_position 
            << " " << OpticksFlags::FlagMask(hit.flag_mask, true)
            << std::endl 
            ;    
        // G4ThreeVector formatter doesnt play well with setw
    }
}


int G4OKTest::rc() const 
{
    return 0 ; 
}


std::string banner(int ievt, char c)
{
    std::string mkr(100, c) ;
    std::stringstream ss ; 
    ss << std::endl ; 
    ss << mkr << " " << std::endl  ; 
    ss << mkr << " " << ievt << std::endl ; 
    ss << mkr << " " << std::endl  ; 
    ss << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}



int main(int argc, char** argv)
{
    int nevt = argc > 1 ? atoi(argv[1]) : 3 ; 

    G4OKTest t(argc, argv); 

    for(int ievt=0 ; ievt < nevt ; ievt++)
    {
       std::cout << banner(ievt,'['); 
       t.collectGensteps(ievt);
       t.propagate(ievt);
       std::cout << banner(ievt,']'); 
    }

    return t.rc() ;
}


