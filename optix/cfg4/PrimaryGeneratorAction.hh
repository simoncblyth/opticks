#pragma once

#include "G4VUserPrimaryGeneratorAction.hh"
#include "globals.hh"

class G4VPrimaryGenerator ;
class G4Event;
class G4SingleParticleSource ;

class RecorderBase ; 

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
  public:
    PrimaryGeneratorAction(RecorderBase*);
    virtual ~PrimaryGeneratorAction();
  private:
    void init();
  public:
    virtual void GeneratePrimaries(G4Event*);
    G4VPrimaryGenerator* MakeGenerator(unsigned int n);
  private:
    G4VPrimaryGenerator*  m_generator ;
    RecorderBase*         m_recorder ;  

};

inline PrimaryGeneratorAction::PrimaryGeneratorAction(RecorderBase* recorder)
 : 
   G4VUserPrimaryGeneratorAction(), 
   m_generator(NULL),
   m_recorder(recorder)
{
   init();

}
