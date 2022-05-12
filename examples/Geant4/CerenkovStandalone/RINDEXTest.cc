
#include <cassert>
#include <iostream>
#include <iomanip>


#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include "OpticksUtil.hh"
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
    a(OpticksUtil::LoadArray(path)),
    rindex( a ? OpticksUtil::MakeProperty(a) : nullptr)
{
    std::cout << "loaded from " << path << std::endl ;  
    assert( a ); 
    assert( rindex ); 
}


void RINDEXTest::g4_line_lookup(double nm0, double nm1, double nm_step)
{
    std::cout << "RINDEXTest::g4_line_lookup" << std::endl ; 
    unsigned count = 0 ; 
    for(double wavelength=nm0 ; wavelength < nm1 ; wavelength += nm_step )
    {
        double energy = h_Planck*c_light/(wavelength*nm) ; 
        double value = rindex->Value(energy); 
        v.push_back(wavelength); 
        v.push_back(value); 

        if(count % 100 == 0 ) std::cout 
            << " count " << std::setw(6) << count 
            << " wavelength " << std::setw(10) << std::fixed << std::setprecision(4) << wavelength
            << " energy/eV " << std::setw(10) << std::fixed << std::setprecision(4) << energy/eV 
            << " value " << std::setw(10) << std::fixed << std::setprecision(4) << value
            << std::endl 
            ;

        count += 1 ; 
    }
}


void RINDEXTest::save(const char* reldir)
{
   if( v.size() > 0 ) 
   {
       // creates reldir if needed
       std::string path = OpticksUtil::prepare_path( FOLD, reldir, "photons.npy" ); 
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

    std::cout << " t.a " << t.a->sstr() << std::endl ; 

    double dscale = 1e6 ; 
    t.a->pdump<double>(argv[0], dscale ); 



    return 0 ;
}
