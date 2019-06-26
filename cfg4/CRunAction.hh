#pragma once

#include "G4UserRunAction.hh"
#include "plog/Severity.h"

class OpticksHub ; 

class CRunAction : public G4UserRunAction
{
        static const plog::Severity LEVEL ; 
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
