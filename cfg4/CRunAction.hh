#pragma once

#include "G4UserRunAction.hh"

class OpticksHub ; 

class CRunAction : public G4UserRunAction
{
    public:
        CRunAction(OpticksHub* hub);
        virtual ~CRunAction();
    public:
        void BeginOfRunAction(const G4Run*);
        void   EndOfRunAction(const G4Run*); 
    private:
        OpticksHub*  m_hub ; 
        unsigned     m_count ; 

};
