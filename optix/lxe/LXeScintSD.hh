#ifndef LXeScintSD_h
#define LXeScintSD_h 1

#include "LXeScintHit.hh"

#include "G4VSensitiveDetector.hh"

class G4Step;
class G4HCofThisEvent;

class LXeScintSD : public G4VSensitiveDetector
{
  public:

    LXeScintSD(G4String name);
    virtual ~LXeScintSD();

    virtual void Initialize(G4HCofThisEvent* );
    virtual G4bool ProcessHits(G4Step* aStep, G4TouchableHistory* );
    virtual void EndOfEvent(G4HCofThisEvent* );
    virtual void clear();
    virtual void DrawAll();
    virtual void PrintAll();
 
  private:

    LXeScintHitsCollection* fScintCollection;
 
};

#endif
