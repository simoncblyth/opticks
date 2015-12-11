#ifndef LXePMTHit_h
#define LXePMTHit_h 1

#include "G4VHit.hh"
#include "G4THitsCollection.hh"
#include "G4Allocator.hh"
#include "G4ThreeVector.hh"
#include "G4LogicalVolume.hh"
#include "G4Transform3D.hh"
#include "G4RotationMatrix.hh"
#include "G4VPhysicalVolume.hh"

#include "tls.hh"

class G4VTouchable;

class LXePMTHit : public G4VHit
{
  public:
 
    LXePMTHit();
    virtual ~LXePMTHit();
    LXePMTHit(const LXePMTHit &right);

    const LXePMTHit& operator=(const LXePMTHit &right);
    G4int operator==(const LXePMTHit &right) const;

    inline void *operator new(size_t);
    inline void operator delete(void *aHit);
 
    virtual void Draw();
    virtual void Print();

    inline void SetDrawit(G4bool b){fDrawit=b;}
    inline G4bool GetDrawit(){return fDrawit;}

    inline void IncPhotonCount(){fPhotons++;}
    inline G4int GetPhotonCount(){return fPhotons;}

    inline void SetPMTNumber(G4int n) { fPmtNumber = n; }
    inline G4int GetPMTNumber() { return fPmtNumber; }

    inline void SetPMTPhysVol(G4VPhysicalVolume* physVol){this->fPhysVol=physVol;}
    inline G4VPhysicalVolume* GetPMTPhysVol(){return fPhysVol;}

    inline void SetPMTPos(G4double x,G4double y,G4double z){
      fPos=G4ThreeVector(x,y,z);
    }
 
    inline G4ThreeVector GetPMTPos(){return fPos;}

  private:

    G4int fPmtNumber;
    G4int fPhotons;
    G4ThreeVector fPos;
    G4VPhysicalVolume* fPhysVol;
    G4bool fDrawit;

};

typedef G4THitsCollection<LXePMTHit> LXePMTHitsCollection;

extern G4ThreadLocal G4Allocator<LXePMTHit>* LXePMTHitAllocator;

inline void* LXePMTHit::operator new(size_t){
  if(!LXePMTHitAllocator)
      LXePMTHitAllocator = new G4Allocator<LXePMTHit>;
  return (void *) LXePMTHitAllocator->MallocSingle();
}

inline void LXePMTHit::operator delete(void *aHit){
  LXePMTHitAllocator->FreeSingle((LXePMTHit*) aHit);
}

#endif
