#include "PLOG.hh"
#include "NP.hh"
#include "SEvt.hh"

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
    sgs s = {} ; 

    s.index = genstep.size() ;  // 0-based genstep index in event (actually since last reset)  
    s.photons = q.numphoton() ;  
    s.offset = getNumPhoton() ; // number of photons in event before this genstep  (actually single last reset)  
    s.gentype = q.gentype() ; 

    gs.push_back(s) ; 
    genstep.push_back(q) ; 

    return s ; 
}

void SEvt::saveGenstep(const char* dir) const 
{
    unsigned num_gs = genstep.size() ; 
    LOG(LEVEL) << " num_gs " << num_gs << " dir " << dir ; 
    std::cout << " num_gs " << num_gs << " dir " << dir << std::endl  ; 
    if(num_gs > 0)
    {
        NP* gs = NP::Make<float>( num_gs, 6, 4 );  
        gs->read2( (float*)genstep.data() );  
        gs->save(dir, "gs.npy"); 
    }
}




std::string SEvt::desc() const 
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < getNumGenstep() ; i++) ss << gs[i].desc() << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}




