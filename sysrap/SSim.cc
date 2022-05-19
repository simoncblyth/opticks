#include "NPFold.h"
#include "PLOG.hh"
#include "SSim.hh"

const plog::Severity SSim::LEVEL = PLOG::EnvLevel("SSim", "DEBUG"); 
SSim* SSim::INSTANCE = nullptr ; 
SSim* SSim::Get(){ return INSTANCE ; }
SSim* SSim::Load(const char* base)
{
    SSim* sim = new SSim ; 
    sim->load(base);  
    return sim ; 
}

SSim::SSim()
    :
    fold(new NPFold)
{
    INSTANCE = this ; 
}

void SSim::add(const char* k, const NP* a ){ fold->add(k,a);  }
const NP* SSim::get(const char* k) const { return fold->get(k);  }
void SSim::load(const char* base){ fold->load(base);  }
void SSim::save(const char* base) const { fold->save(base); }
std::string SSim::desc() const { return fold->desc() ; }

