#include <cassert>
#include "OPTICKS_LOG.hh"

#include "SSys.hh"
#include "G4PVPlacement.hh"
#include "G4Opticks.hh"
#include "G4OpticksRecorder.hh"
#include "G4OpticksHit.hh"
#include "OpticksFlags.hh"

#include "MockSensorAngularEfficiencyTable.hh"

/**
G4OKTest
===============

This test is intended to provide a way of testing G4Opticks machinery 
without ascending to the level of the experiment.

Running with extra hit info using the way buffer
--------------------------------------------------

::

   OPTICKS_EMBEDDED_COMMANDLINE_EXTRA=--way G4OKTest


Identical Photons from Near Identical Gensteps Issue
-------------------------------------------------------

Because this test uses artifical and almost identical "torch" gensteps that 
differ only in the number of photons this will generate duplicated photons 
for each "event",

In future using curand skipahead WITH_SKIPAHEAD will allow the duplication to be avoided 
but anyhow it is useful to not randomize by default as it then makes problems 
of repeated gensteps easier to notice.  
 

Leak Checking
--------------

::

   G4OKTEST_PROFILE_LEAK_MB=10 NEVT=100 G4OKTest 
   G4OKTEST_PROFILE_LEAK_MB=20 NEVT=100 G4OKTest 

::

    2021-02-15 14:58:43.569 INFO  [9756895] [G4Opticks::finalizeProfile@385] saving time/vm stamps to path $TMP/G4Opticks/tests/G4OpticksProfilePlot.npy
    2021-02-15 14:58:43.569 INFO  [9756895] [G4Opticks::finalizeProfile@386] make plot with: ipython -i ~/opticks/g4ok/tests/G4OpticksProfilePlot.py 
     num_stamp 100 m_profile_leak_mb 20     t0 53871.2 t1 53923.6 dt 52.4023 dt/(num_stamp-1) 0.529317     v0 (MB) 35316.6 v1 (MB) 37296.7 dv 1980.09 dv/(num_stamp-1) 20.0009


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
        virtual ~G4OKTest(); 
    private:
        int  initLog(int argc, char** argv);
        void init();
        void initProfile();
        void initCommandLine();
        void initGeometry();
        void initSensorData();
        void initSensorAngularEfficiency();
        void saveSensorLib() const ;
    public: 
        unsigned getNumGenstepPhotons(int eventID) const ;
        void     collectGensteps(int eventID); 
        void     collectInputPhotons(int eventID, const char* path);
        void     propagate(int eventID); 
        void     checkHits(int eventID) const ; 
        int      rc() const ; 
    private:
        int          m_log ; 
        const char*  m_gdmlpath ; 
        const char*  m_opticksCtrl ; // after junoSD_PMT_v2_Opticks
        bool         m_savegensteps ; 
        bool         m_savehits ; 
        int          m_torchtarget ; 
        G4Opticks*   m_g4ok ; 
        G4OpticksRecorder* m_recorder ; 
        bool         m_debug ; 
        bool         m_snap ; 
        const char*  m_tmpdir ; 
};



const plog::Severity G4OKTest::LEVEL = PLOG::EnvLevel("G4OKTest", "DEBUG"); 

G4OKTest::G4OKTest(int argc, char** argv)
    :
    m_log(initLog(argc, argv)),
    m_gdmlpath(PLOG::instance->get_arg_after("--gdmlpath", NULL)),
    m_opticksCtrl(getenv("OPTICKS_CTRL")),
    m_savegensteps(m_opticksCtrl && strstr(m_opticksCtrl, "savegensteps") != nullptr),
    m_savehits(    m_opticksCtrl && strstr(m_opticksCtrl, "savehits")     != nullptr),
    m_torchtarget(PLOG::instance->get_int_after("--torchtarget", "-1")),
    m_g4ok(new G4Opticks),
    m_recorder(new G4OpticksRecorder),
    m_debug(true),
    m_snap(PLOG::instance->has_arg("--snap")),
    m_tmpdir("$TMP/G4OKTest")
{
    init();
}

G4OKTest::~G4OKTest()
{
    m_g4ok->finalize(); 
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

    Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstruction_Opticks.cc

The origin G4PVPlacement for sensor volumes are provided by Opticks in order 
that the corresponding detector specific sensor identifiers can be 
communicated to Opticks via setSensorData. For JUNO the sensor identifiers
can be simply obtained with G4PVPlacement::GetCopyNo.

When running from cache G4PVPlacement are not available, so the 
opticksTripletIdentifier is used as a standin for the sensor identifier. 

**/
void G4OKTest::init()
{
    initProfile(); 
    initCommandLine(); 
    initGeometry();
    initSensorData();
    initSensorAngularEfficiency();
    if(m_debug) saveSensorLib(); 
    //m_g4ok->snap(m_tmpdir);   // snapping before event upload fails due to invalid context : generate.cu requires sequence_buffer
}


void G4OKTest::initProfile()
{
    m_g4ok->setProfile(true); 
    m_g4ok->setProfileLeakMB(SSys::getenvfloat("G4OKTEST_PROFILE_LEAK_MB", 0.f));  
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
    switch(eventID % 10) 
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
    //unsigned num = num_photons*1000 ; 
    unsigned num = num_photons ; 
    LOG(info) << " eventID " << eventID << " num " << num ; 
    return num ; 
}


void G4OKTest::collectInputPhotons(int eventID, const char* path)
{
    LOG(error) 
        << " eventID " << eventID
        << " path " << path 
        ;

    m_g4ok->setInputPhotons(path); 
}


/**
G4OKTest::collectGensteps
--------------------------

Invokes G4Opticks::collectDefaultTorchStep which uses 
the genstep creation and collection machinery. 

**/

void G4OKTest::collectGensteps(int eventID)
{
    unsigned num_genstep_photons = getNumGenstepPhotons(eventID); 
    int node_index = m_torchtarget ; 
    unsigned originTrackID = 100+eventID ;  // arbitrary setting for debugging 

    m_g4ok->collectDefaultTorchStep(num_genstep_photons, node_index, originTrackID ); 

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
    // looking at the input is cheating, the genstep collection machinery should be able to return this

    LOG(error) 
        << " eventID " << eventID
        << " num_genstep_photons " << num_genstep_photons
        << " num_hit " << num_hit 
        ; 

    //m_g4ok->dumpHits("G4OKTest::propagate"); 

    checkHits(eventID); 

    m_g4ok->reset(); // <--- without reset gensteps just keep accumulating 

    if( m_snap && eventID == 0 )
    {       
        m_g4ok->render_snap();
    }

    LOG(LEVEL) << "]" ; 
}

void G4OKTest::checkHits(int eventID) const 
{
    unsigned num_gensteps = m_g4ok->getNumGensteps(); 
    unsigned num_photons = m_g4ok->getNumPhotons(); 
    unsigned num_hit = m_g4ok->getNumHit(); 
    bool way_enabled = m_g4ok->isWayEnabled() ; 

    LOG(info) 
        << " eventID " << eventID
        << " num_gensteps " << num_gensteps
        << " num_photons " << num_photons
        << " num_hit " << num_hit
        << " way_enabled " << way_enabled 
        << " m_savegensteps " << m_savehits
        << " m_savehits " << m_savehits
        << " m_opticksCtrl " << m_opticksCtrl
        ; 


    const char* dir = nullptr ; 
    if(m_savegensteps)
    {
        m_g4ok->saveGensteps(dir, "gs_", eventID, ".npy" ); 
    }

    if(m_savehits)
    {
        m_g4ok->saveHits(    dir, "ht_", eventID, ".npy" ); 
    }


    G4OpticksHit hit ; 
    G4OpticksHitExtra hit_extra ;
    G4OpticksHitExtra* hit_extra_ptr = way_enabled ? &hit_extra : NULL ;

    for(unsigned i=0 ; i < std::min(num_hit,20u) ; i++)
    {
        m_g4ok->getHit(i, &hit, hit_extra_ptr ); 

        std::cout 
            << std::setw(5) << i 
            << " bnd "  << std::setw(4) << hit.boundary 
            << " sIdx " << std::setw(5) << hit.sensorIndex 
            << " nIdx " << std::setw(5) << hit.nodeIndex 
            << " pIdx " << std::setw(5) << hit.photonIndex 
            << " fMsk " << std::setw(10) << std::hex << hit.flag_mask  << std::dec
            << " sID "  << std::setw(10) << std::hex << hit.sensor_identifier << std::dec
            << " nm "   << std::setw(8) << hit.wavelength 
            << " ns "   << std::setw(8) << hit.time
            << " " << std::setw(20) << OpticksFlags::FlagMask(hit.flag_mask, true)
            ;

        if(hit_extra_ptr)
        {   
            std::cout 
                << " hiy "
                << " tk " << hit_extra_ptr->origin_trackID 
                << " t0 " << hit_extra_ptr->origin_time 
                << " bt " << hit_extra_ptr->boundary_time 
                << " bp " << hit_extra_ptr->boundary_pos 
                ;    
        } 
        std::cout << std::endl ; 
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
    int nevt = SSys::getenvint("G4OKTEST_NEVT", 4 ); 
    const char* input_photons_path = SSys::getenvvar("G4OKTEST_INPUT_PHOTONS_PATH", nullptr) ; 

    G4OKTest t(argc, argv); 
    LOG(info) << "(G4OKTEST_NEVT) nevt " << nevt ; 

    for(int ievt=0 ; ievt < nevt ; ievt++)
    {
       std::cout << banner(ievt,'['); 

       if(input_photons_path)
       {
           t.collectInputPhotons(ievt, input_photons_path);
       }
       else
       {
           t.collectGensteps(ievt);
       }

       t.propagate(ievt);
       //std::cout << banner(ievt,']'); 
    }

    return t.rc() ;
}


