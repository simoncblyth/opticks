#ifndef LXeScintHit_h
#define LXeScintHit_h 1

#include "G4VHit.hh"
#include "G4THitsCollection.hh"
#include "G4Allocator.hh"
#include "G4ThreeVector.hh"
#include "G4LogicalVolume.hh"
#include "G4Transform3D.hh"
#include "G4RotationMatrix.hh"
#include "G4VPhysicalVolume.hh"

#include "tls.hh"

class LXeScintHit : public G4VHit
{
  public:
 
    LXeScintHit();
    LXeScintHit(G4VPhysicalVolume* pVol);
    virtual ~LXeScintHit();
    LXeScintHit(const LXeScintHit &right);
    const LXeScintHit& operator=(const LXeScintHit &right);
    G4int operator==(const LXeScintHit &right) const;

    inline void *operator new(size_t);
    inline void operator delete(void *aHit);
 
    virtual void Draw();
    virtual void Print();

    inline void SetEdep(G4double de) { fEdep = de; }
    inline void AddEdep(G4double de) { fEdep += de; }
    inline G4double GetEdep() { return fEdep; }

    inline void SetPos(G4ThreeVector xyz) { fPos = xyz; }
    inline G4ThreeVector GetPos() { return fPos; }

    inline const G4VPhysicalVolume * GetPhysV() { return fPhysVol; }

  private:
    G4double fEdep;
    G4ThreeVector fPos;
    const G4VPhysicalVolume* fPhysVol;

};

typedef G4THitsCollection<LXeScintHit> LXeScintHitsCollection;

extern G4ThreadLocal G4Allocator<LXeScintHit>* LXeScintHitAllocator;

inline void* LXeScintHit::operator new(size_t)
{
  if(!LXeScintHitAllocator)
      LXeScintHitAllocator = new G4Allocator<LXeScintHit>;
  return (void *) LXeScintHitAllocator->MallocSingle();
}

inline void LXeScintHit::operator delete(void *aHit)
{
  LXeScintHitAllocator->FreeSingle((LXeScintHit*) aHit);
}

#endif
