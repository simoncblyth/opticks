#include "CFG4_BODY.hh"
#include "CPrimaryGeneratorAction.hh"
#include "CSource.hh"


CPrimaryGeneratorAction::CPrimaryGeneratorAction(CSource* generator)
    : 
    G4VUserPrimaryGeneratorAction(), 
    m_generator(generator)
{
}

void CPrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
    m_generator->GeneratePrimaryVertex(event);
}

CPrimaryGeneratorAction::~CPrimaryGeneratorAction()
{
    delete m_generator;
}

