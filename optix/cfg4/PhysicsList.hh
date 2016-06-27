#pragma once

#include "G4VModularPhysicsList.hh"

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"
class CFG4_API PhysicsList : public G4VModularPhysicsList
{
  public:

    PhysicsList();
    virtual ~PhysicsList();

  public:

    // SetCuts()
    virtual void SetCuts();

};
#include "CFG4_TAIL.hh"

