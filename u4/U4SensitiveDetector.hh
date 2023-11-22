#pragma once
/**
U4SensitiveDetector.h
======================

Placeholder SensitiveDetector for standalone running 

::

   g4-;g4-cls G4VSensitiveDetector

**/

#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

#include "U4_API_EXPORT.hh"
#include "G4VSensitiveDetector.hh"

struct U4_API U4SensitiveDetector : public G4VSensitiveDetector
{
    static std::vector<U4SensitiveDetector*>* INSTANCES ;  
    static U4SensitiveDetector* Get(const char* name); 
    static std::string Desc(); 

    U4SensitiveDetector(const char* name); 

    G4bool ProcessHits(G4Step* step, G4TouchableHistory* hist);     
};



inline U4SensitiveDetector* U4SensitiveDetector::Get(const char* name)
{
    U4SensitiveDetector* instance = nullptr ; 
    int count = 0 ; 
    int num_instance = INSTANCES ? INSTANCES->size() : 0 ; 

    for(int i=0 ; i < num_instance ; i++)
    {
        U4SensitiveDetector* sd = (*INSTANCES)[i]; 
        G4String sdn = sd->GetName() ; // CAUTION: RETURNS BY VALUE 
        if( strcmp(sdn.c_str(), name) == 0 ) 
        {
            instance = sd ;   
            count += 1 ; 
        } 
    }
    if( count > 1 ) std::cerr 
        << "U4SensitiveDetector::Get"
        << " counted more than one " << count 
        << " with name " << ( name ? name : "-" )
        << std::endl 
        ;

    return instance ; 
}
inline std::string U4SensitiveDetector::Desc()
{
    int num_instance = INSTANCES ? INSTANCES->size() : 0 ; 
    std::stringstream ss ; 
    ss << "U4SensitiveDetector::Desc"
       << " INSTANCES " << ( INSTANCES ? "YES" : "NO " )
       << " num_instance " << num_instance
       << std::endl 
       ;

    for(int i=0 ; i < num_instance ; i++)
    {
        U4SensitiveDetector* sd = (*INSTANCES)[i]; 
        G4String sdn = sd->GetName() ; // CAUTION: RETURNS BY VALUE 
        ss << std::setw(3) << i << " : " << sdn << std::endl ;   
    }
    std::string str = ss.str() ; 
    return str ; 
}


inline U4SensitiveDetector::U4SensitiveDetector(const char* name)
    :
    G4VSensitiveDetector(name)
{
    if(INSTANCES == nullptr) INSTANCES = new std::vector<U4SensitiveDetector*>() ; 
    INSTANCES->push_back(this) ; 
}

inline G4bool U4SensitiveDetector::ProcessHits(G4Step* , G4TouchableHistory* )
{
    //std::cout << "U4SensitiveDetector::ProcessHits" << std::endl ; 
    return true ; 
}

