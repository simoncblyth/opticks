#pragma once

class G4RunManager ;

struct Ctx ; 
struct SensitiveDetector ;
struct DetectorConstruction ;
class L4Cerenkov ; 
template <typename T> struct PhysicsList ; 
struct PrimaryGeneratorAction ;

struct RunAction ; 
struct EventAction ; 
struct TrackingAction ; 
struct SteppingAction ; 

struct G4
{
    G4(int nev); 
    void beamOn(int nev);

    Ctx*                    ctx ; 
    G4RunManager*            rm ; 
    const char*             sdn ; 
    SensitiveDetector*       sd ; 
    DetectorConstruction*    dc ; 
    PhysicsList<L4Cerenkov>* pl ;
    PrimaryGeneratorAction*  ga ; 
    RunAction*               ra ; 
    EventAction*             ea ; 
    TrackingAction*          ta ; 
    SteppingAction*          sa ; 
}; 


