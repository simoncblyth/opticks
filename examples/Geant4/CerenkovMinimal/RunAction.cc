#include "RunAction.hh"
#include "PLOG.hh"

RunAction::RunAction(Ctx* ctx_) 
   :   
     G4UserRunAction(),
     ctx(ctx_)
{
}
void RunAction::BeginOfRunAction(const G4Run*)
{
    LOG(info) << "." ;
}
void RunAction::EndOfRunAction(const G4Run*)
{
    LOG(info) << "." ;
}

