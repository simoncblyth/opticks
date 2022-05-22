#include "PLOG.hh"
#include "NP.hh"
#include "SEvt.hh"
#include "SEvent.hh"

const plog::Severity SEvt::LEVEL = PLOG::EnvLevel("SEvt", "DEBUG"); 

SEvt* SEvt::INSTANCE = nullptr ; 

SEvt::SEvt(){ INSTANCE = this ; }

SEvt* SEvt::Get(){ return INSTANCE ; }

sgs SEvt::AddGenstep(const quad6& q)
{
    if(INSTANCE == nullptr) std::cout << "FATAL: must instanciate SEvt before SEvt::AddGenstep  " << std::endl ; 
    assert(INSTANCE); 
    return INSTANCE->addGenstep(q); 
}
sgs SEvt::AddGenstep(const NP* a)
{
    if(INSTANCE == nullptr) std::cout << "FATAL: must instanciate SEvt before SEvt::AddGenstep  " << std::endl ; 
    assert(INSTANCE); 
    return INSTANCE->addGenstep(a); 
}


void SEvt::AddCarrierGenstep()
{
    AddGenstep(SEvent::MakeCarrierGensteps());
}
void SEvt::AddTorchGenstep()
{
    AddGenstep(SEvent::MakeTorchGensteps());
}






int SEvt::GetNumPhoton()
{
   return INSTANCE ? INSTANCE->getNumPhoton() : -1 ;
}

NP* SEvt::GetGenstep() 
{
   return INSTANCE ? INSTANCE->getGenstep() : nullptr ;
}

void SEvt::clear()
{
    genstep.clear();
    gs.clear(); 
}

unsigned SEvt::getNumGenstep() const 
{
    assert( genstep.size() == gs.size() ); 
    return genstep.size() ; 
}

unsigned SEvt::getNumPhoton() const 
{
    unsigned tot = 0 ; 
    for(unsigned i=0 ; i < genstep.size() ; i++) tot += genstep[i].numphoton() ; 
    return tot ; 
}

sgs SEvt::addGenstep(const quad6& q)
{
    sgs s = {} ;   // genstep summary struct 

    s.index = genstep.size() ;  // 0-based genstep index in event (actually since last reset)  
    s.photons = q.numphoton() ;  
    s.offset = getNumPhoton() ; // number of photons in event before this genstep  (actually single last reset)  
    s.gentype = q.gentype() ; 

    gs.push_back(s) ; 
    genstep.push_back(q) ; 

    return s ; 
}
/**
SEvt::addGenstep
------------------

The sgs summary struct of the last genstep is returned. 

**/

sgs SEvt::addGenstep(const NP* a)
{
    int num_gs = a ? a->shape[0] : -1 ; 
    assert( num_gs > 0 ); 
    quad6* qq = (quad6*)a->bytes(); 
    sgs s = {} ; 
    for(int i=0 ; i < num_gs ; i++) s = addGenstep(qq[i]) ; 
    return s ; 
}



NP* SEvt::getGenstep() const 
{
    unsigned num_gs = genstep.size() ; 
    NP* a = nullptr ; 
    if(num_gs > 0)
    {
        a = NP::Make<float>( num_gs, 6, 4 );  
        a->read2( (float*)genstep.data() );  
    }
    return a ; 
}

void SEvt::saveGenstep(const char* dir) const 
{
    NP* a = getGenstep(); 
    if(a == nullptr) return ; 
    LOG(LEVEL) << a->sstr() << " dir " << dir ; 
    a->save(dir, "gs.npy"); 
}

std::string SEvt::desc() const 
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < getNumGenstep() ; i++) ss << gs[i].desc() << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}


