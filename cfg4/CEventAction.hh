#pragma once

#include "G4UserEventAction.hh"

class OpticksHub ; 

class CEventAction : public G4UserEventAction {
   public:
      CEventAction(OpticksHub* hub);
      virtual ~CEventAction();
   public:
      void BeginOfEventAction(const G4Event* event);
      void EndOfEventAction(const G4Event* event);
   private:
      OpticksHub* m_hub ;
      unsigned    m_count ; 

};
