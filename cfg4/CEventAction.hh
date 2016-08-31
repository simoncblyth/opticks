#pragma once

#include "G4UserEventAction.hh"

class CEventAction : public G4UserEventAction {
   public:
      CEventAction();
      virtual ~CEventAction();
   public:
      void BeginOfEventAction(const G4Event* event);
      void EndOfEventAction(const G4Event* event);
   private:
      unsigned m_count ; 

};
