#pragma once
/**
SPhoton_Debug.h
================

Include into a compilation unit with::

    #include "SPhoton_Debug.h"
    template<> std::vector<SPhoton_Debug<'A'>> SPhoton_Debug<'A'>::record = {} ;

The template char allows multiple static "instances" to be used simultaneously, 
eg in different implementations that are being compared. The char is used to 
prefix the output filename. 

Also add static Save method within the same compilation unit, eg::

    void junoPMTOpticalModel::Save(const char* fold) // static 
    {
        SPhoton_Debug<'A'>::Save(fold);   
    }

Doing all the direct access to the SPhoton_Debug::record from 
only one compilation unit avoids complications of symbol access and 
allows SPhoton_Debug.h to stay headeronly which allows to void 
SysRap needing to depend on Geant4.  
Then from the main, eg u4/tests/U4PMTFastSimTest.cc invoke the Save::

     76     evt->save();
     77     const char* savedir = evt->getSaveDir();
     78     SFastSim_Debug::Save(savedir);
     79     junoPMTOpticalModel::Save(savedir);
  
Essentially what this is doing is adopting a single home 
for the record any only directly accessing it from there.  

Add entries with::
 
    SPhoton_Debug<'A'> dbg ; 

    dbg.pos = ...
    dbg.normal = ...

    dbg.add()

**/

#include <cstdint>
#include <vector>
#include <string>
#include "G4ThreeVector.hh"
#include "NP.hh"    

template<char N>
struct SPhoton_Debug
{
    static std::vector<SPhoton_Debug> record ;   
    static constexpr const char* NAME = "SPhoton_Debug.npy" ; 
    static constexpr const unsigned NUM_QUAD = 4u ; 

    static int Count(); 
    static std::string Name(); 
    static void Save(const char* dir); 
    void add(); 
    void fill(double value); 

    G4ThreeVector pos ;     // 0
    G4double     time ; 

    G4ThreeVector mom ;     // 1 
    uint64_t     iindex ; 

    G4ThreeVector pol ;     // 2 
    G4double      wavelength ; 

    G4ThreeVector nrm ;     // 3
    G4double      spare ; 

}; 

template<char N>
inline std::string SPhoton_Debug<N>::Name() // static
{
    std::string name ; 
    name += N ; 
    name += '_' ; 
    name += NAME ; 
    return name ; 
}

template<char N>
inline void SPhoton_Debug<N>::Save(const char* dir) // static
{
    std::string name = Name(); 
    std::cout  
        << "SPhoton_Debug::Save"
        << " dir " << dir 
        << " name " << name
        << " num_record " << record.size() 
        << std::endl 
        ;

    if( record.size() > 0) NP::Write<double>(dir, name.c_str(), (double*)record.data(), record.size(), NUM_QUAD, 4 );  
    record.clear(); 
}

template<char N>
inline int SPhoton_Debug<N>::Count()  // static
{
    return record.size() ; 
}

template<char N>
inline void SPhoton_Debug<N>::fill(double value)
{
    for(unsigned i=0 ; i < 4*NUM_QUAD ; i++)  *((double*)&pos + i) = value ; 
}

template<char N>
inline void SPhoton_Debug<N>::add()
{ 
    record.push_back(*this);  
}


