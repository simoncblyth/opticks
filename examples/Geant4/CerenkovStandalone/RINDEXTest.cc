
#include <cassert>
#include <iostream>
#include <iomanip>


#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include "OpticksDebug.hh"
#include "NP.hh"

struct RINDEXTest
{
    static const char* FOLD ; 
    NP*                       a ;  
    G4MaterialPropertyVector* rindex ; 
    RINDEXTest(const char* path) ;  

    void g4_line_lookup(double nm0, double nm1, double nm_step); 
    void save(const char* reldir=nullptr); 

    std::vector<double> v ;  
}; 


const char* RINDEXTest::FOLD = "/tmp/RINDEXTest" ; 

RINDEXTest::RINDEXTest( const char* path )
    :
    a(OpticksDebug::LoadArray(path)),
    rindex( a ? OpticksDebug::MakeProperty(a) : nullptr)
{
    std::cout << "loaded from " << path << std::endl ;  
    assert( a ); 
    assert( rindex ); 
}


void RINDEXTest::g4_line_lookup(double nm0, double nm1, double nm_step)
{
    for(double wavelength=nm0 ; wavelength < nm1 ; wavelength += nm_step )
    {
        double energy = h_Planck*c_light/(wavelength*nm) ; 
        double value = rindex->Value(energy); 

        std::cout 
            << " wavelength " << std::setw(10) << std::fixed << std::setprecision(4) << wavelength
            << " energy/eV " << std::setw(10) << std::fixed << std::setprecision(4) << energy/eV 
            << " value " << std::setw(10) << std::fixed << std::setprecision(4) << value
            << std::endl 
            ;

         v.push_back(wavelength); 
         v.push_back(value); 
    }
}


void RINDEXTest::save(const char* reldir)
{
   if( v.size() > 0 ) 
   {
       // creates reldir if needed
       std::string path = OpticksDebug::prepare_path( FOLD, reldir, "photons.npy" ); 
       std::cout << " saving to " << path << std::endl ; 
       NP::Write( FOLD, reldir, "g4_line_lookup.npy", v.data(), v.size()/2, 2 ); 
   }
}


int main(int argc, char** argv)
{
    const char* path = "GScintillatorLib/LS_ori/RINDEX.npy" ; 

    std::cout 
         << " RINDEX path " << path 
         << std::endl 
         ;

    RINDEXTest t(path); 
    t.g4_line_lookup(80., 800., 0.1); 
    t.save(); 

    return 0 ;
}