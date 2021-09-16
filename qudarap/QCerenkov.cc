#include <sstream>
#include <algorithm>

#include "PLOG.hh"
#include "SSys.hh"
#include "SPath.hh"
#include "scuda.h"

#include "NP.hh"

#include "QUDA_CHECK.h"
#include "QRng.hh"
#include "QCerenkov.hh"
#include "QTex.hh"

#include "QCK.hh"


const plog::Severity QCerenkov::LEVEL = PLOG::EnvLevel("QCerenkov", "INFO"); 

const QCerenkov* QCerenkov::INSTANCE = nullptr ; 
const QCerenkov* QCerenkov::Get(){ return INSTANCE ;  }

/**
For now restrict to handling RINDEX from a single material . 
**/

const char* QCerenkov::DEFAULT_PATH = "$OPTICKS_KEYDIR/GScintillatorLib/LS_ori/RINDEX.npy" ;

NP* QCerenkov::Load(const char* path_)  // static
{
    const char* path = SPath::Resolve(path_);  
    NP* a = NP::Load(path); 
    return a ; 
}

QCerenkov::QCerenkov(const char* path_ )
    :
    path( path_ ? path_ : DEFAULT_PATH ),
    dsrc(Load(path)),
    emn(0.),
    emx(0.),
    rmn(0.),
    rmx(0.),
    src(dsrc->ebyte == 4 ? dsrc : NP::MakeNarrow(dsrc)),
    tex(nullptr)
{
    INSTANCE = this ; 
    init(); 
}


void QCerenkov::init()
{
    dsrc->pscale<double>(1e6, 0u) ; //  change energy scale from MeV to eV,   1.55 to 15.5 eV
    dsrc->minmax<double>(emn, emx, 0u ); 
    dsrc->minmax<double>(rmn, rmx, 1u ); 

    LOG(info) 
        << " emn " << std::setw(10) << std::fixed << std::setprecision(4) << emn 
        << " emx " << std::setw(10) << std::fixed << std::setprecision(4) << emx 
        << " rmn " << std::setw(10) << std::fixed << std::setprecision(4) << rmn 
        << " rmx " << std::setw(10) << std::fixed << std::setprecision(4) << rmx 
        ;

    makeTex(src) ;   
}

std::string QCerenkov::desc() const
{
    std::stringstream ss ; 
    ss << "QCerenkov"
       << " dsrc " << ( dsrc ? dsrc->desc() : "-" )
       << " src " << ( src ? src->desc() : "-" )
       << " tex " << ( tex ? tex->desc() : "-" )
       << " tex " << tex 
       ; 

    std::string s = ss.str(); 
    return s ; 
}


/**
QCerenkov::GetAverageNumberOfPhotons_s2
----------------------------------------

This was prototyped and tested in::

    ~/opticks/ana/ckn.py 
    ~/opticks/examples/Geant4/CerenkovStandalone/G4Cerenkov_modifiedTest.cc


"charge" argument is typically 1. or -1, units of Rfact are (eV cm)^-1 so energies must be in eV 
to give Cerenkov photon yield per mm. Essentially Rfact is : "2*pi*fine_structure_constant/hc", approx::

    In [1]: 2.*np.pi*(1./137.)*(1./1240.)*1e6  # hc ~ 1240 eV nm, nm = 1e-6 mm, fine_structure_const ~ 1/137.   
    Out[1]: 36.9860213514221

To understand the more precise Rfact value see ~/opticks/examples/UseGeant4/UseGeant4.cc and google "Frank Tamm Formula" 

**/



template <typename T>
NP* QCerenkov::GetAverageNumberOfPhotons_s2_(T& emin,  T& emax, const T BetaInverse, const T  charge ) const 
{
    emin = emx ; // start with inverted range
    emax = emn ; 
 
    const T* vv = dsrc->cvalues<T>(); 
    unsigned ni = dsrc->shape[0] ; 
    unsigned nj = dsrc->shape[1] ; 
    assert( nj == 2 && ni > 1 ); 

    T s2integral(0.) ;  

    NP* s2i = NP::Make<T>(ni+1) ; 
    T* s2i_vv = s2i->values<T>(); 

    for(unsigned i=0 ; i < ni - 1 ; i++)
    {
        T en_0 = vv[2*(i+0)+0] ; 
        T en_1 = vv[2*(i+1)+0] ; 

        T ri_0 = vv[2*(i+0)+1] ; 
        T ri_1 = vv[2*(i+1)+1] ; 

        T ecross ;  

        T bin_integral = charge*charge*GetS2Integral<T>( emin, emax, ecross, BetaInverse, en_0, en_1, ri_0, ri_1, false );
        s2i_vv[i] = bin_integral ; 

        s2integral += bin_integral ; 
    }
    return s2i ; 
}
template NP* QCerenkov::GetAverageNumberOfPhotons_s2_<double>(double& emin,  double& emax, const double BetaInverse, const double  charge ) const ;
template NP* QCerenkov::GetAverageNumberOfPhotons_s2_<float>( float& emin,   float&  emax, const float  BetaInverse, const float   charge ) const ;



template <typename T>
T QCerenkov::GetAverageNumberOfPhotons_s2(T& emin,  T& emax, const T BetaInverse, const T  charge ) const 
{
    NP* s2i = GetAverageNumberOfPhotons_s2_<T>(emin, emax, BetaInverse, charge ); 
    T s2integral = s2i->psum<T>(0u); 
    return s2integral ; 
}

template double QCerenkov::GetAverageNumberOfPhotons_s2<double>(double& emin,  double& emax, const double BetaInverse, const double  charge ) const ;
template float  QCerenkov::GetAverageNumberOfPhotons_s2<float>( float& emin,   float&  emax, const float  BetaInverse, const float   charge ) const ;




const double QCerenkov::FINE_STRUCTURE_OVER_HBARC_EVMM = 36.981 ; 



/**
QCerenkov::GetS2Integral_WithCut
----------------------------------

Whether the en_cut and ri_cut are used depends on the en_cut value compared to en_0 and en_1 

When integrating trapezoids (like B,C,D) cut bin partial integrals
are simply done by replacing en_1 with the somewhat smaller en_cut.
Integrating edge triangle integrals need more care as the result depends 
on how en_cut compares with en_cross.::

                                 *
                                /|\
                               / | \
                              /  |  \
                             /   |   \
                            /    |    \
             . .           /     |     \   . .
             . .          /      |      \  . .
             . . *-------*       |       * . .
             . ./|       |       |       |\. .
             . / |       |       |       | \ .
             ./.A|  B    |   C   |   D   |E.\.
     -----0--x-.-1-------2-------3-------4-.-y----5----------
             . .                           . .
             . en_cut                      . en_cross
             .                             .          
             en_cross                      en_cut 


    full A:  (en_1 - en_cross)*s2_1*half

    part A:  0.                                  en_cut <= en_cross 
    part A:  (en_cut - en_cross)*s2_cut*half 

    full C:  (en_1 - en_0)*(s2_0 + s2_1)*half 
    part C:  (en_cut - en_0)*(s2_0 + s2_cut)*half

    full E:  (en_cross - en_0)*s2_0*half
    part E:  (en_cross - en_0)*s2_0*half           en_cut >= en_cross
    part E:  (en_cut - en_0)*(s2_0+s2_cut)*half    en_cut < en_cross       *rhs triangle becomes trapezoid* 


* en_cut turns triangle E into a trapezoid for en_cut < en_cross 
* en_cut can only turn triangle A into a smaller triangle  

**/


template<typename T>
T QCerenkov::GetS2Integral_WithCut( const T BetaInverse, const T en_0, const T en_1 , const T ri_0, const T ri_1, const T en_cut, const T ri_cut ) // static 
{
    enum { NONE, CUT, FULL, PART, ERR };  

    int state = NONE ; 
    if(      en_0 >= en_cut && en_1 >  en_cut )  state = CUT ;   
    else if( en_0 < en_cut  && en_1 <= en_cut )  state = FULL ;  // edges of this bin both below en_cut (or en_1 == en_cut), so full bin integral 
    else if( en_0 < en_cut  && en_cut < en_1 )   state = PART ;  // edges of this bin straddle ecut, so partial bin integral  
    else                                         state = ERR ; 
    
    if( state == ERR || state == NONE )
    {
        LOG(error) 
            << " missed condition ?"
            << " en_0 " << en_0 
            << " en_1 " << en_1 
            << " en_cut " << en_cut 
            ;   
    }

    T s2integral(0.) ;  
    if( state == CUT ) return s2integral ; 
    assert( state == FULL || state == PART ); 


    const T zero(0.) ; 
    const T one(1.) ; 
    const T half(0.5) ; 

    T ct_0 = BetaInverse/ri_0 ; 
    T ct_1 = BetaInverse/ri_1 ; 
    T ct_cut = BetaInverse/ri_cut ; 

    T s2_0 = ( one - ct_0 )*( one + ct_0 ); 
    T s2_1 = ( one - ct_1 )*( one + ct_1 ); 
    T s2_cut = ( one - ct_cut )*( one + ct_cut ); 

    bool cross = s2_0*s2_1 < zero ; 
    T en_cross = cross ? en_0 + (BetaInverse - ri_0)*(en_1 - en_0)/(ri_1 - ri_0) : -one  ;


    if( s2_0 < zero && s2_1 > zero )  // s2 becomes +ve within the bin, eg bin A
    {    
        assert( en_cross > zero ); 

        if( state == FULL )
        {  
            s2integral =  (en_1 - en_cross)*s2_1*half ;     // left triangle 
        }
        else if (state == PART )
        {
            s2integral = en_cut <= en_cross ? zero : (en_cut - en_cross)*s2_cut*half ; 
        }
    }    
    else if( s2_0 >= zero && s2_1 >= zero )   // s2 +ve across full bin 
    {    
        if( state == FULL )
        {  
            s2integral = (en_1 - en_0)*(s2_0 + s2_1)*half ;  
        }
        else if( state == PART )
        {
            s2integral = (en_cut - en_0)*(s2_0 + s2_cut)*half ;  
        }
    }     
    else if( s2_0 > zero && s2_1 < zero )  // s2 becomes -ve within the bin
    {    
        assert( en_cross > zero ); 

        if( state == FULL )
        {  
            s2integral =  (en_cross - en_0)*s2_0*half ;  
        }
        else if( state == PART )
        {
            s2integral = en_cut >= en_cross ? (en_cross - en_0)*s2_0*half : (en_cut - en_0)*(s2_0+s2_cut)*half ;   
        }
    }    
    return s2integral ; 
}

template float QCerenkov::GetS2Integral_WithCut( const float, const float, const float, const float, const float, const float, const float ); 
template double QCerenkov::GetS2Integral_WithCut( const double, const double, const double, const double, const double, const double, const double ); 

/**

See cks:G4Cerenkov_modified::GetAverageNumberOfPhotons_s2

s2 numerical integral using trapezoidal approximation, 

* x,y: s2 zero crossings corresponding to RINDEX(E) == BetaInverse
* B,C,D : trapezoids
* A,E   : edge triangles  

                                 *
                                /|\
                               / | \
                              /  |  \
                             /   |   \
                            /    |    \
                           /     |     \
                          /      |      \
                 *-------*       |       *
                /|       |       |       |\
               / |       |       |       | \
              / A|  B    |   C   |   D   |E \
     -----0--x---1-------2-------3-------4---y----5----------
            /                                 \

* determine x,y energy crossings from BetaInverse-RINDEX(E) crossings, as that is linear 
  so the linear interpolation to give crossings will be precise.
  Using s2 zero crossing not as precise, as non-linear. 


**WARNING : SLIGHTLY INACCURATE WHEN CUT : HOPEFULLY FIXED IN ABOVE _WithCut**

**/

template<typename T>
T QCerenkov::GetS2Integral( T& emin, T& emax, T& en_cross, const T BetaInverse, const T en_0, const T en_1 , const T ri_0, const T ri_1, bool fix_cross ) // static 
{
    const T zero(0.) ; 
    const T one(1.) ; 
    const T half(0.5) ; 
    T s2integral(0.) ;  

    T ct_0 = BetaInverse/ri_0 ; 
    T ct_1 = BetaInverse/ri_1 ; 

    T s2_0 = ( one - ct_0 )*( one + ct_0 ); 
    T s2_1 = ( one - ct_1 )*( one + ct_1 ); 


    bool cross = s2_0*s2_1 < zero ; 
    if( cross )
    {
        if(fix_cross)  // using input en_cross
        {
            assert( en_cross > zero );  
        }
        else           // calulate en_cross for use in calling scope
        {
            en_cross = en_0 + (BetaInverse - ri_0)*(en_1 - en_0)/(ri_1 - ri_0) ; // compute en_cross for this range 
        }
    }
    else               // set en_cross to dummy placeholder, as not used when no crossings ... hmm bit risky 
    {
         en_cross = -one ; 
    }


    if( s2_0 <= zero && s2_1 <= zero )   // entire bin disallowed : no contribution 
    {    
        s2integral = zero ; 
    }    
    else if( s2_0 < zero && s2_1 > zero )  // s2 becomes +ve within the bin
    {    
        assert( en_cross > zero ); 
        s2integral =  (en_1 - en_cross)*s2_1*half ;  

        emin = std::min<T>( emin, en_cross );  
        emax = std::max<T>( emax, en_1 );
    }    
    else if( s2_0 >= zero && s2_1 >= zero )   // s2 +ve across full bin 
    {    
        s2integral = (en_1 - en_0)*(s2_0 + s2_1)*half ;  

        emin = std::min<T>( emin, en_0 );  
        emax = std::max<T>( emax, en_1 );
    }     
    else if( s2_0 > zero && s2_1 < zero )  // s2 becomes -ve within the bin
    {    
        assert( en_cross > zero ); 
        s2integral =  (en_cross - en_0)*s2_0*half ;  

        emin = std::min<T>( emin, en_0 );  
        emax = std::max<T>( emax, en_cross );
    }    
    else 
    {    
        std::cout 
            << "QCerenkov::getS2Integral"
            << " FATAL "
            << " s2_0 " << std::fixed << std::setw(10) << std::setprecision(5) << s2_0 
            << " s2_1 " << std::fixed << std::setw(10) << std::setprecision(5) << s2_1 
            << " en_0 " << std::fixed << std::setw(10) << std::setprecision(5) << en_0 
            << " en_1 " << std::fixed << std::setw(10) << std::setprecision(5) << en_1 
            << " ri_0 " << std::fixed << std::setw(10) << std::setprecision(5) << ri_0 
            << " ri_1 " << std::fixed << std::setw(10) << std::setprecision(5) << ri_1 
            << std::endl
            ;
        assert(0);
    }
    const T Rfact = FINE_STRUCTURE_OVER_HBARC_EVMM ;      // hmm maybe better scale once the total ? 
    return Rfact*s2integral ; 
}


template double QCerenkov::GetS2Integral( double& , double&, double&, const double, const double, const double, const double, const double, bool ) ; 
template float  QCerenkov::GetS2Integral( float& ,  float& , float&,  const float , const float , const float , const float , const float , bool ) ; 




/**
QCerenkov::getS2CutIntegral
-------------------------------------

See ana/s2.py : the full bin integral is done to 
fix en_cross to avoid imprecision arising from 
repeatedly obtaining slightly different en_cross
as the range of the cumulative integral is extended.

Each bin is consided for its contribution to the integral 
by comparison of ecut with the edge energies.
Full contrib for this bin:: 
                
                                   ecut 
                                    |
     +------+---------+=======+-----*------+
                    en_0     en_1   |
                                    | 



Hmm potentially a problem when the ecut chops into the s2 rampup triangle, 
from unhandled conditions ?
      
The contortions to be able to reuse GetS2Integral may be misplaced : 
simpler to consider the cut integral separately ?


**/

template <typename T>
NP* QCerenkov::getS2CutIntegral_( const T BetaInverse, const T ecut ) const 
{
    T emin, emax, en_cross ; 

    const T* vv = dsrc->cvalues<T>(); 
    unsigned ni = dsrc->shape[0] ; 
    unsigned nj = dsrc->shape[1] ; 
    assert( nj == 2 && ni > 1 ); 

    NP* s2i = NP::Make<T>(ni+1);  // indices 0...ni inclusive, ni used for the partial 
    T* s2i_vv = s2i->values<T>();  

    T s2integral(0.) ;  
    for(unsigned i=0 ; i < ni - 1 ; i++)
    {
        T en_0 = vv[2*(i+0)+0] ; 
        T en_1 = vv[2*(i+1)+0] ; 

        T ri_0 = vv[2*(i+0)+1] ; 
        T ri_1 = vv[2*(i+1)+1] ; 

        if( en_0 == ecut )                 // no contribution from this bin 
        {
        }
        else if( en_0 < ecut && en_1 <= ecut)  // edges of this bin both below ecut (or en_1 == ecut), so full bin integral 
        {
            T bin_integral =  GetS2Integral<T>( emin, emax, en_cross, BetaInverse, en_0, en_1, ri_0, ri_1, false );  
            s2i_vv[i] = bin_integral ;    
            s2integral += bin_integral ; 
        }
        else if ( en_0 < ecut && ecut < en_1 )  // edges of this bin straddle ecut, so partial bin integral  
        {
            // full bin integral to get en_cross 
            T full = GetS2Integral<T>( emin, emax, en_cross, BetaInverse, en_0, en_1, ri_0, ri_1, false  ); 

            // partial bin integral using the fixed en_cross ; 
            T ri_cut = dsrc->interp<T>(ecut) ; 
            T partial  = GetS2Integral<T>( emin, emax, en_cross, BetaInverse, en_0, ecut, ri_0, ri_cut, true );

            // hmm: s2 can be increasing or decreasing here ... so need to handle the two cases ? 

            s2i_vv[ni] = partial ; 
   
            bool is_partial = partial <= full ;  // 
            if(!is_partial)
            {
                LOG(info)
                    << " is_partial " << is_partial
                    << " en_cross " << std::fixed << std::setw(10) << std::setprecision(4) << en_cross
                    << " BetaInverse " << std::fixed << std::setw(10) << std::setprecision(4) << BetaInverse 
                    << " en_0 " << std::fixed << std::setw(10) << std::setprecision(4) << en_0
                    << " en_1 " << std::fixed << std::setw(10) << std::setprecision(4) << en_1
                    << " ecut " << std::fixed << std::setw(10) << std::setprecision(4) << ecut
                    << " ri_cut " << std::fixed << std::setw(10) << std::setprecision(4) << ri_cut
                    << " full " << std::fixed << std::setw(10) << std::setprecision(4) << full 
                    << " partial " << std::fixed << std::setw(10) << std::setprecision(4) << partial
                    << " full-partial " << std::fixed << std::setw(10) << std::setprecision(4) << full-partial
                    ; 
            }
            //assert( is_partial ); 

            s2integral += partial ; 
        }
        else if(  en_0 > ecut && en_1 > ecut )  // bin fully above the cut, so no contribution
        {
        }
        else
        {
            LOG(warning) 
                << " missing  ?"
                << " en_0 " << en_0 
                << " en_1 " << en_1 
                << " ecut " << ecut 
                ;   
        }


    }
    return s2i ; 
}     


template NP* QCerenkov::getS2CutIntegral_( const double, const double  ) const ; 
template NP* QCerenkov::getS2CutIntegral_( const float, const float  ) const ; 

template <typename T>
T QCerenkov::getS2CutIntegral( const T BetaInverse, const T ecut ) const 
{
    NP* s2i = getS2CutIntegral_<T>(BetaInverse, ecut); 
    T s2integral = s2i->psum<T>(0u); 
    return s2integral ; 
}  
template double QCerenkov::getS2CutIntegral( const double, const double  ) const ; 
template float  QCerenkov::getS2CutIntegral( const float, const float  ) const ; 



template <typename T> NP* QCerenkov::getS2Integral_WithCut_( const T BetaInverse, const T en_cut ) const 
{
    const T* vv = dsrc->cvalues<T>(); 
    unsigned ni = dsrc->shape[0] ; 
    unsigned nj = dsrc->shape[1] ; 
    assert( nj == 2 && ni > 1 ); 

    NP* s2i = NP::Make<T>(ni);
    T* s2i_vv = s2i->values<T>();  

    T ri_cut = dsrc->interp<T>(en_cut) ; 

    for(unsigned i=0 ; i < ni - 1 ; i++)
    {
        T en_0 = vv[2*(i+0)+0] ; 
        T en_1 = vv[2*(i+1)+0] ; 

        T ri_0 = vv[2*(i+0)+1] ; 
        T ri_1 = vv[2*(i+1)+1] ; 

        s2i_vv[i] = GetS2Integral_WithCut<T>(BetaInverse, en_0, en_1, ri_0, ri_1, en_cut, ri_cut );
    }
    return s2i ; 
}
 
template NP* QCerenkov::getS2Integral_WithCut_( const double, const double  ) const ; 
template NP* QCerenkov::getS2Integral_WithCut_( const float, const float  ) const ; 



template <typename T>
T QCerenkov::getS2Integral_WithCut( const T BetaInverse, const T en_cut ) const 
{
    NP* s2i = getS2Integral_WithCut_<T>(BetaInverse, en_cut); 
    T s2integral = s2i->psum<T>(0u); 
    return s2integral ; 
}  
template double QCerenkov::getS2Integral_WithCut( const double, const double  ) const ; 
template float  QCerenkov::getS2Integral_WithCut( const float, const float  ) const ; 






/**
QCerenkov::getS2CumulativeIntegrals
-------------------------------------

Returns array of shape (nx, 2) with energies and 
cululative integrals up to those energies. 
Note that the energy range isadapted to correspond to the
permissable range of Cerenkov for the BetaInverse to 
make best use of the available bins. 

The cumulative integral values are not normalized. 

**/

template <typename T>
NP* QCerenkov::getS2CumulativeIntegrals( const T BetaInverse, unsigned nx ) const 
{
    T emin ; 
    T emax ; 
    T charge = T(1.) ;  

    T avgNumPhotons = GetAverageNumberOfPhotons_s2<T>(emin, emax, BetaInverse, charge ); 
    if(avgNumPhotons <= 0. ) return nullptr ; 

    NP* s2c = NP::Make<T>(nx, 2) ; 
    T* cc = s2c->values<T>();

    NP* full_s2i = GetAverageNumberOfPhotons_s2_<T>(emin, emax, BetaInverse, charge );

    LOG(debug) 
        << " BetaInverse " << std::setw(10) << std::fixed << std::setprecision(4) << BetaInverse
        << " emin " << std::setw(10) << std::fixed << std::setprecision(4) << emin
        << " emax " << std::setw(10) << std::fixed << std::setprecision(4) << emax
        << " avgNumPhotons " << std::setw(10) << std::fixed << std::setprecision(4) << avgNumPhotons
        << " avgNumPhotons*1e6 " << std::setw(10) << std::fixed << std::setprecision(4) << avgNumPhotons*1e6
        ;

    const NP* edom = NP::Linspace<T>( emin, emax, nx ); 
    const T* ee = edom->cvalues<T>(); 

    T last_ecut = 0. ; 
    T last_s2integral = 0. ; 
    NP* last_s2i = nullptr ; 
 
    for(unsigned i=0 ; i < nx ; i++)
    {
        T ecut = ee[i] ; 
        
        //NP* s2i = getS2CutIntegral_<T>( BetaInverse, ecut ); 
        NP* s2i = getS2Integral_WithCut_<T>( BetaInverse, ecut ); 
        T s2integral = s2i->psum<T>(0u); 

        cc[2*i+0] = ecut ;   
        cc[2*i+1] = s2integral ; 

        if( i == nx - 1 )
        {
            last_ecut = ecut ; 
            last_s2integral = s2integral ; 
            last_s2i = s2i ; 
        }
        else
        {
            s2i->clear(); 
        }
    }

    T diff = std::abs(avgNumPhotons - last_s2integral); 
    bool close = diff < 1e-6 ; 
    if(!close)
    {
        std::cout << "NP::DumpCompare a:full_s2i b:last_s2i " << std::endl;  
        NP::DumpCompare<double>( full_s2i, last_s2i, 0u, 0u, 1e-6 ); 

        std::cout 
            << " QCerenkov::getS2CumulativeIntegrals "
            << " BetaInverse " << std::setw(10) << std::fixed << std::setprecision(4) << BetaInverse
            << " avgNumPhotons " << std::setw(10) << std::fixed << std::setprecision(4) << avgNumPhotons   
            << " last_ecut " << std::setw(10) << std::fixed << std::setprecision(4) << last_ecut 
            << " last_s2integral " << std::setw(10) << std::fixed << std::setprecision(4) << last_s2integral
            << " diff " << std::setw(10) << std::fixed << std::setprecision(4) << diff
            << " close " << close 
            << std::endl 
            ;
    }
    //assert(close);   // last cut integral should be close to the avgNumPhotons

    return s2c ; 
}

template NP* QCerenkov::getS2CumulativeIntegrals( const double BetaInverse, unsigned nx ) const ; 
template NP* QCerenkov::getS2CumulativeIntegrals( const float  BetaInverse, unsigned nx ) const ; 


/**
NP* QCerenkov::getS2CumulativeIntegrals
-----------------------------------------

BetaInverse beyond which get no photons could be used to determine 
the bis range, avoiding a large chunk of the output array containing just zeros.

**/

template <typename T>
NP* QCerenkov::getS2CumulativeIntegrals( const NP* bis, unsigned nx ) const 
{
    unsigned ni = bis->shape[0] ; 
    const T* bb = bis->cvalues<T>(); 
    unsigned nj = nx ; 

    NP* s2c = NP::Make<T>(ni, nj, 2) ; 
    for(unsigned i=0 ; i < ni ; i++)
    {
        const T BetaInverse = bb[i] ; 
        NP* s2c_one = getS2CumulativeIntegrals<T>(BetaInverse, nx ); 
 
        if(s2c_one == nullptr)
        {
            LOG(debug) 
                << " s2c_one NULL " 
                << " i " << i 
                << " ni " << ni 
                << " BetaInverse " << std::setw(10) << std::fixed << std::setprecision(4) << BetaInverse 
                ; 
            continue; 
        }
        unsigned s2c_one_bytes = s2c_one->arr_bytes() ;  
        memcpy( s2c->bytes() + i*s2c_one_bytes, s2c_one->bytes(), s2c_one_bytes ); 
        s2c_one->clear(); 
    }
    return s2c ; 
}

template NP* QCerenkov::getS2CumulativeIntegrals<double>( const NP* , unsigned ) const ; 
template NP* QCerenkov::getS2CumulativeIntegrals<float>(  const NP* , unsigned ) const ; 


/**
QCK QCerenkov::makeICDF     
--------------------------

ny 
    number BetaInverse values "height"
nx
    number of energy domain values "width"

**/

template <typename T>
QCK<T> QCerenkov::makeICDF( unsigned ny, unsigned nx ) const 
{
    NP* bis = NP::Linspace<T>( 1. , rmx,  ny ) ;  
    NP* s2c = getS2CumulativeIntegrals<T>( bis, nx ); 
    NP* s2cn = s2c->copy(); 
    s2cn->divide_by_last<T>(); 

    QCK<T> icdf ; 

    icdf.rindex = dsrc ; 
    icdf.bis = bis ;  
    icdf.s2c = s2c ;
    icdf.s2cn = s2cn ; 

    return icdf ; 
}
template QCK<double> QCerenkov::makeICDF<double>( unsigned , unsigned ) const ; 
template QCK<float>  QCerenkov::makeICDF<float>(  unsigned , unsigned ) const ; 






void QCerenkov::makeTex(const NP* src)
{
    LOG(LEVEL) << desc() ; 

    unsigned nx = 0 ; 
    unsigned ny = 0 ; 


    //tex = new QTex<float>(nx, ny, src->getValuesConst(), filterMode ) ; 

    //tex->setHDFactor(HDFactor(dsrc)); 
    //tex->uploadMeta(); 

    LOG(LEVEL)
        << " src " << src->desc()
        << " nx (width) " << nx
        << " ny (height) " << ny
        ;

}

extern "C" void QCerenkov_check(dim3 numBlocks, dim3 threadsPerBlock, unsigned width, unsigned height  ); 
extern "C" void QCerenkov_lookup(dim3 numBlocks, dim3 threadsPerBlock, cudaTextureObject_t texObj, quad4* meta, float* lookup, unsigned num_lookup, unsigned width, unsigned height  ); 

void QCerenkov::configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height )
{
    threadsPerBlock.x = 512 ; 
    threadsPerBlock.y = 1 ; 
    threadsPerBlock.z = 1 ; 
 
    numBlocks.x = (width + threadsPerBlock.x - 1) / threadsPerBlock.x ; 
    numBlocks.y = (height + threadsPerBlock.y - 1) / threadsPerBlock.y ;
    numBlocks.z = 1 ; 

    LOG(LEVEL) 
        << " width " << std::setw(7) << width 
        << " height " << std::setw(7) << height 
        << " width*height " << std::setw(7) << width*height 
        << " threadsPerBlock"
        << "(" 
        << std::setw(3) << threadsPerBlock.x << " " 
        << std::setw(3) << threadsPerBlock.y << " " 
        << std::setw(3) << threadsPerBlock.z << " "
        << ")" 
        << " numBlocks "
        << "(" 
        << std::setw(3) << numBlocks.x << " " 
        << std::setw(3) << numBlocks.y << " " 
        << std::setw(3) << numBlocks.z << " "
        << ")" 
        ;
}

void QCerenkov::check()
{
    unsigned width = tex->width ; 
    unsigned height = tex->height ; 

    LOG(LEVEL)
        << " width " << width
        << " height " << height
        ;

    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    configureLaunch( numBlocks, threadsPerBlock, width, height ); 
    QCerenkov_check(numBlocks, threadsPerBlock, width, height );  

    cudaDeviceSynchronize();
}


NP* QCerenkov::lookup()
{
    unsigned width = tex->width ; 
    unsigned height = tex->height ; 
    unsigned num_lookup = width*height ; 

    LOG(LEVEL)
        << " width " << width
        << " height " << height
        << " lookup " << num_lookup
        ;

    NP* out = NP::Make<float>(height, width ); 

    float* out_ = out->values<float>(); 
    lookup( out_ , num_lookup, width, height ); 

    return out ; 
}

void QCerenkov::lookup( float* lookup, unsigned num_lookup, unsigned width, unsigned height  )
{
    LOG(LEVEL) << "[" ; 
    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    configureLaunch( numBlocks, threadsPerBlock, width, height ); 
    
    size_t size = width*height*sizeof(float) ; 
  
    LOG(LEVEL) 
        << " num_lookup " << num_lookup
        << " width " << width 
        << " height " << height
        << " size " << size 
        << " tex->texObj " << tex->texObj
        << " tex->meta " << tex->meta
        << " tex->d_meta " << tex->d_meta
        ; 

    float* d_lookup = nullptr ;  
    QUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_lookup ), size )); 

    QCerenkov_lookup(numBlocks, threadsPerBlock, tex->texObj, tex->d_meta, d_lookup, num_lookup, width, height );  

    QUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( lookup ), d_lookup, size, cudaMemcpyDeviceToHost )); 
    QUDA_CHECK( cudaFree(d_lookup) ); 

    cudaDeviceSynchronize();

    LOG(LEVEL) << "]" ; 
}

void QCerenkov::dump( float* lookup, unsigned num_lookup, unsigned edgeitems  )
{
    LOG(LEVEL); 
    for(unsigned i=0 ; i < num_lookup ; i++)
    {
        if( i < edgeitems || i > num_lookup - edgeitems )
        std::cout 
            << std::setw(6) << i 
            << std::setw(10) << std::fixed << std::setprecision(3) << lookup[i] 
            << std::endl 
            ; 
    }
}

