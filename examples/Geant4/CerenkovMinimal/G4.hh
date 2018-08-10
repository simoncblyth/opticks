#pragma once

struct Ctx ; 
class G4RunManager ;
struct DetectorConstruction ;
class L4Cerenkov ; 
template <typename T> struct PhysicsList ; 
struct PrimaryGeneratorAction ;
struct EventAction ; 
struct TrackingAction ; 
struct SteppingAction ; 

struct G4
{
    G4(); 
    void beamOn(int nev);

    Ctx*                    ctx ; 
    G4RunManager*            rm ; 
    DetectorConstruction*    dc ; 
    PhysicsList<L4Cerenkov>* pl ;
    PrimaryGeneratorAction*  ga ; 
    EventAction*             ea ; 
    TrackingAction*          ta ; 
    SteppingAction*          sa ; 
}; 


