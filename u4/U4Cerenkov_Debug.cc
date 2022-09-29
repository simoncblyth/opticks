#include "U4Cerenkov_Debug.hh"
#include "U4Debug.hh"
#include "NP.hh"    
#include "PLOG.hh"

const plog::Severity U4Cerenkov_Debug::LEVEL = PLOG::EnvLevel("U4Cerenkov_Debug", "debug" ); 
std::vector<U4Cerenkov_Debug> U4Cerenkov_Debug::record = {} ;

void U4Cerenkov_Debug::EndOfEvent(int eventID)
{
    const char* dir = U4Debug::GetSaveDir(eventID);
    LOG(LEVEL) << " dir " << dir << " num_record " << record.size() ;
    std::cout  << " dir " << dir << " num_record " << record.size() << std::endl ;
    if( record.size() > 0) NP::Write<double>(dir, NAME, (double*)record.data(), record.size(), NUM_QUAD, 4 );  
    record.clear(); 
}

void U4Cerenkov_Debug::add()
{
    LOG(LEVEL) << "num_record " << record.size() ;
    if(record.size() < LIMIT) record.push_back(*this); 
}


