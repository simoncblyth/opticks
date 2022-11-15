#include "U4RecorderTest.h"
#include "SEvt.hh"

struct U4PMTFastSimTest
{
    G4VUserPhysicsList*        phy ; 
    G4RunManager*              run ; 
    U4RecorderTest*            rec ; 

    U4PMTFastSimTest(); 
    void beamOn(int n); 

    virtual ~U4PMTFastSimTest(); 
};

U4PMTFastSimTest::U4PMTFastSimTest()
    :
    phy((G4VUserPhysicsList*)new U4Physics),
    run(new G4RunManager),
    rec(nullptr)
{
    run->SetUserInitialization(phy); 
    rec = new U4RecorderTest(run) ;  
}

void U4PMTFastSimTest::beamOn(int n)
{
    run->BeamOn(n); 
}

U4PMTFastSimTest::~U4PMTFastSimTest(){ delete rec ; }

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SEvt evt ; 
    SEvt::AddTorchGenstep(); 

    U4PMTFastSimTest t ;  
    //t.beamOn(1); 

    return 0 ; 
}

