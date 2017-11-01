#include "CFG4_BODY.hh"
#include "CPrimaryGeneratorAction.hh"
#include "CSource.hh"


CPrimaryGeneratorAction::CPrimaryGeneratorAction(CSource* source)
    : 
    G4VUserPrimaryGeneratorAction(), 
    m_source(source)
{
}

void CPrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
    m_source->GeneratePrimaryVertex(event);
}

CPrimaryGeneratorAction::~CPrimaryGeneratorAction()
{
    delete m_source;
}

