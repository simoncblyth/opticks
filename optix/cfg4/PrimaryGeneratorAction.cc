#include "PrimaryGeneratorAction.hh"
#include "OpSource.hh"

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
    m_generator->GeneratePrimaryVertex(event);
}

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
    delete m_generator;
}

