#include "G4String.hh"
#include "U4RecorderTest.h"
//#include "PMTFastSim.hh"

//struct DetectorConstruction ; 
//class HamamatsuR12860PMTManager ;
class G4LogicalVolume ;
//class junoPMTOpticalModel ;

struct U4PMTFastSimTest
{
    bool                       verbose ;  
    G4String                   label ; 
    //DetectorConstruction*      dc ; 
    //HamamatsuR12860PMTManager* mgr ; 
    const char*                geom ; 
    //G4LogicalVolume*           lv ; 
    //junoPMTOpticalModel*       pom ; 
    G4VUserPhysicsList*        phy ; 
    G4RunManager*              run ; 
    U4RecorderTest*            rec ; 

    U4PMTFastSimTest(); 
    void init(); 
    virtual ~U4PMTFastSimTest(); 
};


//#include "DetectorConstruction.hh"
//#include "HamamatsuR12860PMTManager.hh"
//#include "junoPMTOpticalModel.hh"

#include "Layr.h"
#include "SDirect.hh"
#include "NP.hh"


U4PMTFastSimTest::U4PMTFastSimTest()
    :
    verbose(getenv("VERBOSE")!=nullptr),
    label("U4PMTFastSimTest"),
    //dc(nullptr),
    //mgr(nullptr),
    geom(getenv("U4PMTFastSimTest_GEOM")),
    //lv(PMTFastSim::GetLV(geom)),
    //pom(nullptr),
    phy(nullptr),
    run(nullptr),
    rec(nullptr)
{
    init();
}

U4PMTFastSimTest::~U4PMTFastSimTest()
{   
    delete rec ; 
}

void U4PMTFastSimTest::init()
{

    /*
    std::stringstream coutbuf;
    std::stringstream cerrbuf;
    {   
        cout_redirect out_(coutbuf.rdbuf());
        cerr_redirect err_(cerrbuf.rdbuf());
   
        dc = new DetectorConstruction ; 
    }   
    std::string out = coutbuf.str();
    std::string err = cerrbuf.str();
    std::cout << OutputMessage("U4PMTFastSimTest::init" , out, err, verbose );

    mgr = new HamamatsuR12860PMTManager(label) ;
    lv = mgr->getLV() ; 
    pom = mgr->pmtOpticalModel ; 
    */


    phy = (G4VUserPhysicsList*)new U4Physics ; 
    run = new G4RunManager ; 
    run->SetUserInitialization(phy); 

    rec = new U4RecorderTest(run) ;  
    run->BeamOn(1); 
}

int main(int argc, char** argv)
{
    std::cout << argv[0] << std::endl ; 
    U4PMTFastSimTest t ;  
    return 0 ; 
}

