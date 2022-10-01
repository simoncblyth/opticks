#include "U4Hit_Debug.hh"
#include "U4Debug.hh"
#include "NP.hh"    
#include "SLOG.hh"

const plog::Severity U4Hit_Debug::LEVEL = SLOG::EnvLevel("U4Hit_Debug", "debug" ); 
std::vector<U4Hit_Debug> U4Hit_Debug::record = {} ;

void U4Hit_Debug::Save(const char* dir)
{
    LOG(LEVEL) << " dir " << dir << " num_record " << record.size() ;
    std::cout  << " dir " << dir << " num_record " << record.size() << std::endl ;
    assert( NUM_QUAD == 1u ); 
    if( record.size() > 0) NP::Write<int>(dir, NAME, (int*)record.data(), record.size(), 4 );  
    record.clear(); 
}

void U4Hit_Debug::add()
{
    LOG(LEVEL) << "num_record " << record.size() ;
    if(record.size() < LIMIT) record.push_back(*this); 
}



