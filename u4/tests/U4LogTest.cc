
#include "scuda.h"

#include "G4Log.hh"
#include "OPTICKS_LOG.hh"
#include "SSys.hh"
#include "spath.h"
#include "NP.hh"


struct U4LogTest
{
    static std::string Prod( const char* v_, double v, const char* vsc_, double sc, const char* fvsc_ ); 

    static const char* FOLD ; 
    unsigned ni ; 
    static constexpr const unsigned NJ = 5 ; 

    U4LogTest(); 
    std::string one(double u); 
    void one_( double4& dd, double d ); 
    void scan(); 

}; 


const char* U4LogTest::FOLD = spath::Resolve("$TMP/U4LogTest");


U4LogTest::U4LogTest()
    :
    ni(SSys::getenvunsigned("U4LogTest_ni", 1000001))
{
}


std::string U4LogTest::Prod( const char* v_, double v, const char* vsc_, double sc, const char* fvsc_  )
{
    std::stringstream ss ; 
    ss << " " << std::setw(10) << v_   << " " << std::setw(10) << std::fixed << std::setprecision(7) << v
       << " " << std::setw(10) << vsc_ << " " << std::setw(10) << std::fixed << std::setprecision(7) << v*sc
       << " " << std::setw(10) << fvsc_ << " " << std::setw(10) << std::fixed << std::setprecision(7) << float(v)*float(sc) 
       ; 
    std::string s = ss.str(); 
    return s ; 
}


std::string U4LogTest::one( double u )
{
    double4 dd ; 
    one_(dd, u );  

    double sc = 1e7 ; 
    std::stringstream ss ; 
    ss 
        << " u  " << std::setw(10) << std::fixed << std::setprecision(7) << u << std::endl 
        << Prod("d0",dd.x,"d0*sc", sc, "f(d0)*f(sc)" ) << std::endl
        << Prod("f0",dd.y,"f0*sc", sc, "f(f0)*f(sc)" ) << std::endl
        << Prod("d4",dd.z,"d4*sc", sc, "f(d4)*f(sc)" ) << std::endl
        << Prod("f4",dd.w,"f4*sc", sc, "f(f4)*f(sc)" ) << std::endl
        ; 

    std::string s = ss.str(); 
    return s ; 
}

void U4LogTest::one_( double4& dd, double d )
{
    float  f = float(d) ; 
    double d0 = -1.*std::log( d ); 
    float  f0 = -1.f*std::log( f ); 
    double d4 = -1.*G4Log( d ) ; 
    float  f4 = -1.f*G4Logf( f ) ; 

    dd.x = d0 ; 
    dd.y = f0 ; 
    dd.z = d4 ; 
    dd.w = f4 ; 
}

void U4LogTest::scan()
{
    NP* a = NP::Make<double>(ni, NJ ); 
    double* aa = a->values<double>();  

    for(unsigned i=0 ; i < ni ; i++)
    {
        double d =  double(i)/double(ni-1) ; 

        double4 dd ; 
        one_(dd, d );  
       
        aa[NJ*i+0] = d ; 
        aa[NJ*i+1] = dd.x ; 
        aa[NJ*i+2] = dd.y ; 
        aa[NJ*i+3] = dd.z ;
        aa[NJ*i+4] = dd.w ; 
    }
    a->save(FOLD, "scan.npy") ; 
}

int main(int argc, char** argv)
{
    U4LogTest t ; 
    double u = SSys::getenvdouble("U", 0.) ; 
    if( u > 0. ) std::cout << t.one(u) ; 
    else t.scan(); 

    return 0 ;  
}
