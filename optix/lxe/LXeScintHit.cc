#include "LXeScintHit.hh"
#include "G4ios.hh"
#include "G4VVisManager.hh"
#include "G4Colour.hh"
#include "G4VisAttributes.hh"
#include "G4LogicalVolume.hh"
#include "G4VPhysicalVolume.hh"

G4ThreadLocal G4Allocator<LXeScintHit>* LXeScintHitAllocator=0;


LXeScintHit::LXeScintHit() : fEdep(0.), fPos(0.), fPhysVol(0) {}


LXeScintHit::LXeScintHit(G4VPhysicalVolume* pVol) : fPhysVol(pVol) {}


LXeScintHit::~LXeScintHit() {}


LXeScintHit::LXeScintHit(const LXeScintHit &right) : G4VHit()
{
  fEdep = right.fEdep;
  fPos = right.fPos;
  fPhysVol = right.fPhysVol;
}


const LXeScintHit& LXeScintHit::operator=(const LXeScintHit &right){
  fEdep = right.fEdep;
  fPos = right.fPos;
  fPhysVol = right.fPhysVol;
  return *this;
}


G4int LXeScintHit::operator==(const LXeScintHit&) const{
  return false;
  //returns false because there currently isnt need to check for equality yet
}


void LXeScintHit::Draw() {}


void LXeScintHit::Print() {}
