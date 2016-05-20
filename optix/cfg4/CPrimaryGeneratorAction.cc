#include "CPrimaryGeneratorAction.hh"
#include "CSource.hh"

void CPrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
    m_generator->GeneratePrimaryVertex(event);
}

CPrimaryGeneratorAction::~CPrimaryGeneratorAction()
{
    delete m_generator;
}

