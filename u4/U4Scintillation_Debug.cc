#include "U4Scintillation_Debug.hh"
#include "U4Debug.hh"
#include "NP.hh"    
#include "PLOG.hh"

const plog::Severity U4Scintillation_Debug::LEVEL = PLOG::EnvLevel("U4Scintillation_Debug", "debug" ); 
std::vector<U4Scintillation_Debug> U4Scintillation_Debug::record = {} ;

void U4Scintillation_Debug::Save(const char* dir)
{
    LOG(LEVEL) << " dir " << dir << " num_record " << record.size() ;
    std::cout  << " dir " << dir << " num_record " << record.size() << std::endl ;
    if( record.size() > 0) NP::Write<double>(dir, NAME, (double*)record.data(), record.size(), NUM_QUAD, 4 );  
    record.clear(); 
}

void U4Scintillation_Debug::add()
{
    LOG(LEVEL) << "num_record " << record.size() ;
    if(record.size() < LIMIT) record.push_back(*this); 
}

