#pragma once

#include "G4VModularPhysicsList.hh"
#include "globals.hh"

class PhysicsList: public G4VModularPhysicsList
{
  public:

    PhysicsList();
    virtual ~PhysicsList();

  public:

    // SetCuts()
    virtual void SetCuts();

};

