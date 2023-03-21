#pragma once

/**
U4AppTest (formerly U4SimulateTest)
--------------------------------------

* U4App *app* instanciated in ctor, that in turn instanciates U4Recorder 
* NB methods are not inlined because it only makes sense to include this once 
  into the main of for example u4/tests/U4SimulateTest 

**/

class G4RunManager ; 
class G4VUserPhysicsList ;
struct U4App ; 
#include "ssys.h"

struct U4AppTest
{
    static G4RunManager* InitRunManager(G4VUserPhysicsList* phy);  
    G4VUserPhysicsList*        phy ; 
    G4RunManager*              run ; 
    U4App*                     app ; 

    U4AppTest(); 
    void BeamOn(); 
    virtual ~U4AppTest(){ delete app ; }
};

G4RunManager* U4AppTest::InitRunManager(G4VUserPhysicsList* phy)
{
    G4RunManager* run = new G4RunManager ; 
    run->SetUserInitialization(phy) ; 
    return run ; 
}

U4AppTest::U4AppTest()
    :
    phy((G4VUserPhysicsList*)new U4Physics),
    run(InitRunManager(phy)),
    app(new U4App(run))
{
}

void U4AppTest::BeamOn()
{
    run->BeamOn(ssys::getenvint("BeamOn",1)); 
}



