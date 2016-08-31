#pragma once

#include "G4UserRunAction.hh"

class CRunAction : public G4UserRunAction
{
    public:
        CRunAction();
        virtual ~CRunAction();
    public:
        void BeginOfRunAction(const G4Run*);
        void   EndOfRunAction(const G4Run*); 
    private:
        unsigned m_count ; 

};
