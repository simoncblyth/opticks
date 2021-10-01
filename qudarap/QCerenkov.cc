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


const plog::Severity QCerenkov::LEVEL = PLOG::EnvLevel("QCerenkov", "DEBUG"); 

const QCerenkov* QCerenkov::INSTANCE = nullptr ; 
const QCerenkov* QCerenkov::Get(){ return INSTANCE ;  }


const unsigned QCerenkov::SPLITBIN_PAYLOAD_SIZE = 8 ; 
const unsigned QCerenkov::UPPERCUT_PAYLOAD_SIZE = 3 ; 


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

    T en_a = -1. ; 
    T ri_a = -1. ; 
    T en_b = -1. ; 
    T ri_b = -1. ; 

    NP* s2i = NP::Make<T>(ni) ;   // hmm: top value will always be zero 
    T* s2i_vv = s2i->values<T>(); 
    bool dump = false ; 

    for(unsigned i=0 ; i < ni - 1 ; i++)
    {
        T en_0 = vv[2*(i+0)+0] ; 
        T en_1 = vv[2*(i+1)+0] ; 

        T ri_0 = vv[2*(i+0)+1] ; 
        T ri_1 = vv[2*(i+1)+1] ; 

        T bin_integral = charge*charge*GetS2Integral_WithCut<T>( emin, emax, BetaInverse, en_0, en_1, ri_0, ri_1, en_a, ri_a, en_b, ri_b, dump );

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



template <typename T>
NP* QCerenkov::getAverageNumberOfPhotons_s2( const NP* bis ) const   
{
    T emin ; 
    T emax ; 
    T charge = T(1.) ;  

    unsigned ni = bis->shape[0] ; 
    NP* avph = NP::Make<T>(ni, 4 ); 
    T* avph_v = avph->values<T>(); 

    for(unsigned i=0 ; i < ni ; i++)
    {
        const T BetaInverse = bis->get<T>(i) ; 
        T avgNumPhotons = GetAverageNumberOfPhotons_s2<T>(emin, emax, BetaInverse, charge ); 

        avph_v[4*i+0] = BetaInverse ; 
        avph_v[4*i+1] = emin  ; 
        avph_v[4*i+2] = emax ; 
        avph_v[4*i+3] = avgNumPhotons ; 
    }
    return avph ; 
}

template NP* QCerenkov::getAverageNumberOfPhotons_s2<float>( const NP* ) const ; 
template NP* QCerenkov::getAverageNumberOfPhotons_s2<double>( const NP* ) const ; 










const double QCerenkov::FINE_STRUCTURE_OVER_HBARC_EVMM = 36.981 ; 

const char* QCerenkov::NONE_ = "NONE" ; 
const char* QCerenkov::UNCUT_ = "UNCUT" ; 
const char* QCerenkov::SUB_ = "SUB" ; 
const char* QCerenkov::CUT_ = "CUT" ; 
const char* QCerenkov::FULL_ = "FULL" ; 
const char* QCerenkov::PART_ = "PART" ; 
const char* QCerenkov::ERR_ = "ERR" ; 

const char* QCerenkov::State(int state)
{
    const char* s = nullptr ; 
    switch(state)
    {
       case NONE:  s = NONE_  ; break ; 
       case SUB:   s = SUB_   ; break ; 
       case UNCUT: s = UNCUT_ ; break ; 
       case CUT:   s = CUT_   ; break ; 
       case FULL:  s = FULL_  ; break ; 
       case PART:  s = PART_  ; break ; 
       case ERR:   s = ERR_   ; break ; 
    }
    return s ; 
}



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
             . en_b                        . en_cross
             .                             .          
             en_cross                      en_b   


    full A:  (en_1 - en_cross)*s2_1*half

    part A:  0.                                  en_cut <= en_cross 
    part A:  (en_cut - en_cross)*s2_cut*half 

    full C:  (en_1 - en_0)*(s2_0 + s2_1)*half 
    part C:  (en_cut - en_0)*(s2_0 + s2_cut)*half

    full E:  (en_cross - en_0)*s2_0*half
    part E:  (en_cross - en_0)*s2_0*half           en_cut >= en_cross
    part E:  (en_cut - en_0)*(s2_0+s2_cut)*half    en_cut < en_cross       *rhs triangle becomes trapezoid* 

* en_b (formerly known as en_cut) is used for multi-bin integrals up to en_b(en_cut)
* en_b turns triangle E into a trapezoid for en_b < en_cross 
* en_b can only turn triangle A into a smaller triangle  

* subsequently the en_a argument was added to provide SUB-bin integrals between en_a and en_b 
  which are both required to be within en_0 en_1 of a single bin 



              en_0                 en_1 
               +--------------------+
               |                    |
         bbbbbb|                    |                     ( en_0 > en_b  && en_1 > en_b )     -> CUT     bin above the cut 
               |                    |
               |                    |bbbbb                ( en_0 < en_b  && en_1 <= en_b )    -> FULL    bin below the cut                      
               |                    |
               bbbbbbbbbbbbbbbbbbbbb|                     ( en_0 <= en_b && en_b < en_1 )     -> PART           
               |                    |
               aaaaaaaabbbbbbbbbbbbbb                     ( en_a >= en_0 && en_b <= en_1 )     -> SUB 
               |                    |
               |                    |
               |                    |
               +--------------------+

**/

template<typename T>
T QCerenkov::GetS2Integral_WithCut( T& emin, T& emax, T BetaInverse, T en_0, T en_1 , T ri_0, T ri_1, T en_a, T ri_a, T en_b, T ri_b, bool dump ) // static 
{

    int state = NONE ; 
    if(      en_b <= 0.  )                   state = UNCUT ;   
    else if( en_a >= en_0 && en_b <= en_1 )  state = SUB ;   // SUB-bin integral with sub-range en_a:en_b entirely within(including edge) range of big bin en_0:en_1
    else if( en_0 >  en_b && en_1 >  en_b )  state = CUT ;   
    else if( en_0 <  en_b && en_1 <= en_b )  state = FULL ;  // edges of this bin both below en_b (or en_1 == en_b), so full bin integral 
    else if( en_0 <= en_b && en_b <  en_1 )  state = PART ;  // edges of this bin straddle ecut, so partial bin integral  
    else                                     state = ERR ; 
    
    if( state == ERR || state == NONE ) LOG(error) << " missed condition ?" << " en_0 " << en_0 << " en_1 " << en_1 << " en_a " << en_a << " en_b " << en_b ;  
 
    const T zero(0.) ; 
    if( state == CUT ) return zero ; 
    assert( state == FULL || state == PART || state == UNCUT || state == SUB ); 
    if(state == SUB) assert( en_a >= en_0 && en_b <= en_1 ); 

    const T one(1.) ; 
    const T half(0.5) ; 
    T s2integral(0.) ;  

    T ct_0 = BetaInverse/ri_0 ; 
    T ct_1 = BetaInverse/ri_1 ; 
    T ct_a = en_a > 0 ? BetaInverse/ri_a : -1 ; 
    T ct_b = en_b > 0 ? BetaInverse/ri_b : -1 ; 
    T s2_0 = ( one - ct_0 )*( one + ct_0 ); 
    T s2_1 = ( one - ct_1 )*( one + ct_1 );
    T s2_a = en_a > 0 ? ( one - ct_a )*( one + ct_a ) : -1 ; 
    T s2_b = en_b > 0 ? ( one - ct_b )*( one + ct_b ) : -1 ; 

    if(dump) std::cout 
        << "QCerenkov::GetS2Integral_WithCut"
        << " en_0 " << en_0 
        << " en_1 " << en_1 
        << " en_a " << en_a 
        << " en_b " << en_b 
        << " s2_0 " << s2_0 
        << " s2_1 " << s2_1 
        << " s2_a " << s2_a
        << " s2_b " << s2_b
        << " state " << State(state)
        << " BetaInverse " << BetaInverse
        << std::endl
        ;


    bool cross = s2_0*s2_1 < zero ; 
    T en_cross = cross ? en_0 + (BetaInverse - ri_0)*(en_1 - en_0)/(ri_1 - ri_0) : -one  ;


    if( s2_0 < zero && s2_1 > zero )  // s2 becomes +ve within the bin, eg bin A : lhs triangle 
    {    
        assert( en_cross > zero ); 

        if( state == FULL || state == UNCUT )
        {  
            s2integral =  (en_1 - en_cross)*s2_1*half ;     // left triangle 

            emin = std::min<T>( emin, en_cross );  
            emax = std::max<T>( emax, en_1 );
        }
        else if (state == PART )
        {
            s2integral = en_b <= en_cross ? zero : (en_b - en_cross)*s2_b*half ; 

            emin = std::min<T>( emin, en_b <= en_cross ? emin : en_cross );  
            emax = std::max<T>( emax, emax );
        }
        else if (state == SUB )
        {
            /**
             Three possibilites for en_a and en_b wrt en_cross : both below, straddling, both above 
                                  /
                 en_a     en_b   / 
               +---+-------+----*---------------+         zero        
              en_0            en_cross         en_1
            
                                  /
                          en_a   /  en_b                  triangle from en_cross to en_b 
               +-----------+----*----+----------+                
              en_0             en_cross        en_1
            
                                  /
                                 /  en_a  en_b            trapezoid independent of en_cross
               +----------------*----+-----+----+                
              en_0           en_cross          en_1
            
            **/

            if( en_a <= en_cross && en_b <= en_cross )
            {
                 s2integral = zero ; 
            }
            else
            {
                 if( en_a <= en_cross && en_b >= en_cross )
                 {
                     s2integral = (en_b - en_cross)*s2_b*half ; 
                     emin = std::min<T>( emin, en_cross ) ; 
                     emax = std::min<T>( emax, en_b ) ; 
                 }
                 else
                 {
                     s2integral = (en_b - en_a)*(s2_a + s2_b)*half ; 
                     emin = std::min<T>( emin, en_a ) ; 
                     emax = std::min<T>( emax, en_b ) ; 
                 }
            }
        }
    }    
    else if( s2_0 >= zero && s2_1 >= zero )   // s2 +ve across full bin 
    {    
        if( state == FULL || state == UNCUT )
        {  
            s2integral = (en_1 - en_0)*(s2_0 + s2_1)*half ;  

            emin = std::min<T>( emin, en_0 );  
            emax = std::max<T>( emax, en_1 );
        }
        else if( state == PART )
        {
            s2integral = (en_b - en_0)*(s2_0 + s2_b)*half ;  
            if(dump) std::cout << " s2 +ve across full bin " << " s2integral " << s2integral << std::endl ; 
            emin = std::min<T>( emin, en_0 );  
            emax = std::max<T>( emax, en_b );
        }
        else if (state == SUB )
        {
            s2integral = (en_b - en_a)*(s2_a + s2_b)*half ;  
            emin = std::min<T>( emin, en_a );  
            emax = std::max<T>( emax, en_b );
        }
    }     
    else if( s2_0 > zero && s2_1 < zero )  // s2 becomes -ve within the bin, eg bin E :  rhs triangle 
    {    
        assert( en_cross > zero ); 

        if( state == FULL || state == UNCUT )
        {  
            s2integral =  (en_cross - en_0)*s2_0*half ;  

            emin = std::min<T>( emin, en_0 );  
            emax = std::max<T>( emax, en_cross );

        }
        else if( state == PART )
        {
            s2integral = en_b >= en_cross ? (en_cross - en_0)*s2_0*half : (en_b - en_0)*(s2_0+s2_b)*half ;   

            emin = std::min<T>( emin, en_0 );  
            emax = std::max<T>( emax, en_b >= en_cross ? en_cross : en_b );
        }
        else if (state == SUB )
        {
            /**
             Three possibilites for en_a and en_b wrt en_cross : both below, straddling, both above 

                             \
                              \
                 en_a     en_b \
               +---+-------+----*---------------+         trapezoid independent of en_cross        
              en_0            en_cross         en_1
            
                             \
                              \
                          en_a \    en_b                  triangle from en_a to en_cross 
               +-----------+----*----+----------+                
              en_0             en_cross        en_1
            
                             \
                              \
                               \    en_a  en_b            zero
               +----------------*----+-----+----+                
              en_0           en_cross          en_1
            
            **/
            if( en_a <= en_cross && en_b <= en_cross )
            {
                 s2integral = (en_b - en_a)*(s2_a + s2_b)*half ; 
                 emin = std::min<T>( emin, en_a ) ; 
                 emax = std::min<T>( emax, en_b ) ; 
            }
            else
            {
                 if( en_a <= en_cross && en_b >= en_cross )
                 {
                     s2integral = (en_cross - en_a)*s2_a*half ; 
                     emin = std::min<T>( emin, en_a ) ; 
                     emax = std::min<T>( emax, en_cross ) ; 
                 }
                 else
                 {
                     s2integral = zero ; 
                 }
            }
        }
    }    
    const T Rfact = FINE_STRUCTURE_OVER_HBARC_EVMM ;  
    return Rfact*s2integral ; 
}

template float QCerenkov::GetS2Integral_WithCut( float& , float&,    float,  float,  float,  float,  float,  float,  float, float, float, bool  ); 
template double QCerenkov::GetS2Integral_WithCut( double&, double&,  double,  double,  double,  double,  double,  double,  double, double, double, bool  ); 


/**
QCerenkov::getS2Integral_WithCut_
------------------------------------

For debugging purposes the contributions to the integral from each rindex bin are returned in an array.

**/

template <typename T> NP* QCerenkov::getS2Integral_WithCut_( T& emin, T& emax, T BetaInverse, T en_a, T en_b, bool dump ) const 
{
    const T* vv = dsrc->cvalues<T>(); 
    unsigned ni = dsrc->shape[0] ; 
    unsigned nj = dsrc->shape[1] ; 
    assert( nj == 2 && ni > 1 ); 

    NP* s2i = NP::Make<T>(ni);
    T* s2i_vv = s2i->values<T>();  

    T ri_a = en_a > 0 ? dsrc->interp<T>(en_a) : -1 ; 
    T ri_b = en_b > 0 ? dsrc->interp<T>(en_b) : -1 ; 

    for(unsigned i=0 ; i < ni - 1 ; i++)
    {
        T en_0 = vv[2*(i+0)+0] ; 
        T en_1 = vv[2*(i+1)+0] ; 

        T ri_0 = vv[2*(i+0)+1] ; 
        T ri_1 = vv[2*(i+1)+1] ; 

        s2i_vv[i] = GetS2Integral_WithCut<T>(emin, emax, BetaInverse, en_0, en_1, ri_0, ri_1, en_a, ri_a, en_b, ri_b, dump );
    }
    return s2i ; 
}
 
template NP* QCerenkov::getS2Integral_WithCut_( double&, double&, double, double, double, bool  ) const ; 
template NP* QCerenkov::getS2Integral_WithCut_( float&,  float&,  float,  float,  float,  bool  ) const ; 



template <typename T>
T QCerenkov::getS2Integral_WithCut( T& emin, T& emax, T BetaInverse, T en_a, T en_b, bool dump ) const 
{
    NP* s2i = getS2Integral_WithCut_<T>(emin, emax, BetaInverse, en_a, en_b, dump ); 
    T s2integral = s2i->psum<T>(0u); 
    return s2integral ; 
}  
template double QCerenkov::getS2Integral_WithCut( double&, double&, double, double, double, bool  ) const ; 
template float  QCerenkov::getS2Integral_WithCut( float&,  float&,  float, float, float, bool   ) const ; 


/**
QCerenkov::getS2Integral_splitbin
-------------------------------------

Returns *s2c* array of shape (s2_edges, PAYLOAD_SIZE(=8)) 
with the last payload entry being s2integral.

The s2 value of evaluated at en_cut and s2integral 
is the cululative integral up to en_cut.  The first 
value of s2integral is zero. 

* this approach leads to lots of zero bins because of the fixed full energy range 
* would be simple to restrict to big bins with contributions allowing mul to be increased at the same cost 
  by greatly reducing zeros

Why smaller sub-bins are needed ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Objective is to effectively provide an inverse-CDF such that
a random number lookup yields an energy. 
As s2 is piecewise linear with energy(from the piecewise linear RINDEX) 
the cumulative integral will be piecewise parabolic. 
Hence linear interpolation on that parabolic is not going to be a very 
good approximation, so want to make the binning smaller than the
rindex bins. 

Simplification : by bin-splitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Avoid complexity and imprecision arising from differences between 
rindex energy edges and the smaller sub bins by adopting approach of ana/edges.py:divide_bins
and more directly NP::MakeDiv 

* there is no need for the sub-bins to all be equally spaced, the motivation for 
  the sub-bins is to profit from the piecewise linear nature of s2 which means that 
  s2 can be linearly interpolated with very small errors unlike the parabolic
  cumulative integral for which linear interpolation is a poor approximation 

* splitting the edges into sub-bins allows to simply loop within each big rindex 
  bin saving the partials off into the cumulative integral array   


not easy to avoid lots of s2c zero bins in this _SplitBin approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* with _UpperCut the number of energy bins is imposed to 
  be equal for all BetaInverse and different Cerenkov permissable 
  energy ranges are used for each BetaInverse

* to do something similar here in _SplitBin would need to have the 
  number of energy bins different for each BetaInverse, 
  for example including only sub-bins within big-bins with contributions
  
* would need to NP::Combine ragged individual BetaInverse s2c 
* complexity seems not worth it, especially as s2c/s2cn are just 
  intermediaries on way to creating fixed 0:1 domain icdf array 
  that is the input to creating the GPU texture 

* that icdf is inherently regular fixed domain as created by NP::pdomain 
  inversion of s2cn across 0->1 range  


prev/next cumulative approach of Geant4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prev/next approach commonly used in Geant4 BuildThePhysicsTable 
for cumulative integration could slightly reduce computation as en_1 ri_1 s2_1 
becomes en_0 ri_0 s2_0 for next bin. 

Maybe that helps avoid non-monotonic ?


See Also
~~~~~~~~~~~~

* ana/piecewise.py for some whacky parabolic Ideas

TODO:

* create the icdf float4 payload array, by baking NP::pdomain results from 0->1 for each BetaInverse 
* check lookups using it still OK chi2 with sampling 
* turn into GPU texture and check again, will probably need the hd_factor trick from 
  X4Scintillation::CreateGeant4InterpolatedInverseCDF in the energy extremes where the CDF 
  tends to get very flat (and hence ICDF is very steep)
  

**/


template <typename T>
NP* QCerenkov::getS2Integral_SplitBin(const T BetaInverse, unsigned mul, bool dump ) const 
{
    T emin, emax ; 
    T charge = T(1.) ;  
    T avgNumPhotons = GetAverageNumberOfPhotons_s2<T>(emin, emax, BetaInverse, charge ); 
    if(avgNumPhotons <= 0. ) return nullptr ; 

    LOG(debug)
       << " emin " << std::setw(10) << std::fixed << std::setprecision(4) << emin
       << " emax " << std::setw(10) << std::fixed << std::setprecision(4) << emax
       ;

    const T* ri_v = dsrc->cvalues<T>(); 

    unsigned ri_ni = dsrc->shape[0] ; 
    unsigned ri_nj = dsrc->shape[1] ; 
    assert( ri_nj == 2 && ri_ni > 1 );

    unsigned ns = 1+mul ;                // sub divisions of one bin 
    unsigned s2_edges = getNumEdges_SplitBin<T>(mul); 

    NP* s2c = NP::Make<T>(s2_edges, SPLITBIN_PAYLOAD_SIZE) ; // number of values/edges is one more than bins
    T* s2c_v = s2c->values<T>(); 

    std::vector<unsigned> idxs ;  
    unsigned idx = 0 ; 
    for(unsigned p=0 ; p < SPLITBIN_PAYLOAD_SIZE ; p++) s2c_v[SPLITBIN_PAYLOAD_SIZE*idx + p] = ( p == 0 || p == 1) ? emn : 0. ; 
    // artifical cumulative integral zero for 1st entry 
    idxs.push_back(idx); 

    T s2integral = 0. ; 

    for(unsigned i=0 ; i < ri_ni - 1 ; i++)
    {
        T en_0 = ri_v[2*(i+0)+0] ; 
        T ri_0 = ri_v[2*(i+0)+1] ; 

        T en_1 = ri_v[2*(i+1)+0] ; 
        T ri_1 = ri_v[2*(i+1)+1] ; 

        T en_01 = en_1 - en_0 ; 

        if(dump) std::cout 
             << "QCerenkov::getS2Integral_splitbin"
             << " i " << std::setw(3) << i 
             << " ns " << std::setw(3) << ns 
             << " en_0 " << std::setw(10) << std::fixed << std::setprecision(4) << en_0
             << " en_1 " << std::setw(10) << std::fixed << std::setprecision(4) << en_1
             << std::endl 
             ;

        // starting from s=1 skips first sub-edge as that is the same as the last edge from prior bin
        for(unsigned s=1 ; s < 1+mul  ; s++)   
        { 
            T en_a = en_0 + en_01*T(s-1)/T(mul) ;  // mul=1 ->s=1 only : en_a=en_0      
            T en_b = en_0 + en_01*T(s+0)/T(mul) ;  // mul=1 ->s=1 only : en_b=en_1     

            T ri_a = en_a > 0. ? dsrc->interp<T>(en_a) : -1 ; 
            T ri_b = en_b > 0. ? dsrc->interp<T>(en_b) : -1 ; 

            T s2_a = getS2<T>( BetaInverse, en_a ); 
            T s2_b = getS2<T>( BetaInverse, en_b ); 

            T sub = GetS2Integral_WithCut<T>(emin, emax, BetaInverse, en_0, en_1, ri_0, ri_1, en_a, ri_a, en_b, ri_b, dump );

            s2integral += sub  ; 

            unsigned idx = i*mul + s ; 
            idxs.push_back(idx); 
            assert( idx < s2_edges ); 

            s2c_v[SPLITBIN_PAYLOAD_SIZE*idx + 0 ] = en_b ;        // 1st payload slot must be "domain" for NP::pdomain lookup ... 
            s2c_v[SPLITBIN_PAYLOAD_SIZE*idx + 1 ] = en_a ;        // en_b en_a inversion needed .. as 1st entry is auto set to emn 
            s2c_v[SPLITBIN_PAYLOAD_SIZE*idx + 2 ] = ri_a ;  
            s2c_v[SPLITBIN_PAYLOAD_SIZE*idx + 3 ] = ri_b ;  
            s2c_v[SPLITBIN_PAYLOAD_SIZE*idx + 4 ] = s2_a ;  
            s2c_v[SPLITBIN_PAYLOAD_SIZE*idx + 5 ] = s2_b ;  
            s2c_v[SPLITBIN_PAYLOAD_SIZE*idx + 6 ] = sub ;  
            s2c_v[SPLITBIN_PAYLOAD_SIZE*idx + 7 ] = s2integral ;   // last payload slot must be "value" for NP::pdomain lookup
        }
    }

    for(unsigned i=0 ; i < idxs.size() ; i++) assert( idxs[i] == i );  // check idx has it covered 
    return s2c ; 
} 


template NP* QCerenkov::getS2Integral_SplitBin( const double, unsigned, bool ) const ; 
template NP* QCerenkov::getS2Integral_SplitBin( const float,  unsigned, bool ) const ; 


template<typename T>
unsigned QCerenkov::getNumEdges_SplitBin(unsigned mul ) const 
{
    unsigned ri_ni = dsrc->shape[0] ; 
    unsigned ri_bins = ri_ni - 1 ;       // number of bins is one less than number of values 
    unsigned s2_bins = ri_bins*mul ; 
    return s2_bins + 1 ; 
}





template unsigned QCerenkov::getNumEdges_SplitBin<float>(unsigned ) const ; 
template unsigned QCerenkov::getNumEdges_SplitBin<double>(unsigned ) const ; 

/**
QCerenkov::getS2Integral_SplitBin
-----------------------------------

Note that this is requiring the same number of s2c s2_edges for all BetaInverse.
Directly using the s2c to yield the icdf would remove that requirement.

**/

template <typename T>
NP* QCerenkov::getS2Integral_SplitBin( const NP* bis, unsigned mul, bool dump) const 
{
    unsigned ni = bis->shape[0] ; 
    unsigned s2_edges = getNumEdges_SplitBin<T>(mul); 
    unsigned nj = s2_edges ; 

    NP* s2c = NP::Make<T>(ni, nj, SPLITBIN_PAYLOAD_SIZE) ; 

    LOG(LEVEL) << "[ creating s2c " << s2c->sstr() ; 

    for(unsigned i=0 ; i < ni ; i++)
    {
        const T BetaInverse = bis->get<T>(i) ; 
        NP* s2c_one = getS2Integral_SplitBin<T>(BetaInverse, mul, dump ); 
 
        if(s2c_one == nullptr)
        {
            LOG(info) 
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
    LOG(LEVEL) << "] creating s2c " << s2c->sstr() ; 
    return s2c ; 
}

template NP* QCerenkov::getS2Integral_SplitBin<float>(  const NP*, unsigned, bool) const ; 
template NP* QCerenkov::getS2Integral_SplitBin<double>( const NP*, unsigned, bool) const ; 




template <typename T>
T QCerenkov::getS2( const T BetaInverse, const T en ) const 
{
    const T one(1.) ; 
    T ri = dsrc->interp<T>(en) ;  // linear interpolation of the rindex values at edges 
    T ct = BetaInverse/ri ; 
    T s2 = ( one - ct )*( one + ct ); 
    return s2 ; 
}

template double QCerenkov::getS2( const double, const double ) const ; 
template float  QCerenkov::getS2( const float, const float ) const ; 



/**
QCerenkov::getS2Integral_UpperCut
-------------------------------------

Returns array of shape (nx, 2) with energies and 
cululative integrals up to those energies. 
Note that the energy range isadapted to correspond to the
permissable range of Cerenkov for the BetaInverse to 
make best use of the available bins. 

The cumulative integral values are not normalized. 


HMM: getting slightly non-monotonic again in some regions, 
idea to avoid that is to avoid repeating the calc just save the cumulative 
result in the one pass

**/

template <typename T>
NP* QCerenkov::getS2Integral_UpperCut( const T BetaInverse, unsigned nx ) const   
{
    T emin ; 
    T emax ; 
    T charge = T(1.) ;  

    T avgNumPhotons = GetAverageNumberOfPhotons_s2<T>(emin, emax, BetaInverse, charge ); 
    if(avgNumPhotons <= 0. ) return nullptr ; 


    NP* s2c = NP::Make<T>(nx, UPPERCUT_PAYLOAD_SIZE) ; 
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
    bool dump = false ; 

    // This is doing the full integral with increasing ecut=en_b at each turn 
    // so it slightly grows, problem with this is that its very repetitive and hence slow
    // and also it risks going slightly non-monotonic from numerical imprecision.
    //
 
    for(unsigned i=0 ; i < nx ; i++)
    {
        T en_a = -1. ; 
        T en_b = ee[i] ; 
        
        NP* s2i = getS2Integral_WithCut_<T>(emin, emax, BetaInverse, en_a, en_b, dump ); 

        T s2integral = s2i->psum<T>(0u); 
        T s2_b = getS2<T>( BetaInverse, en_b ); 

        cc[UPPERCUT_PAYLOAD_SIZE*i+0] = en_b ;         // domain qty must be 1st payload entry for NP::pdomain
        cc[UPPERCUT_PAYLOAD_SIZE*i+1] = s2_b ; 
        cc[UPPERCUT_PAYLOAD_SIZE*i+2] = s2integral ;   // qty to be normalized needs to be in the last payload slot for NP::divide_by_last  

        if( i == nx - 1 )
        {
            last_ecut = en_b ; 
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
            << " QCerenkov::getS2Integral_UpperCut "
            << " BetaInverse " << std::setw(10) << std::fixed << std::setprecision(4) << BetaInverse
            << " avgNumPhotons " << std::setw(10) << std::fixed << std::setprecision(4) << avgNumPhotons   
            << " last_ecut " << std::setw(10) << std::fixed << std::setprecision(4) << last_ecut 
            << " last_s2integral " << std::setw(10) << std::fixed << std::setprecision(4) << last_s2integral
            << " diff " << std::setw(10) << std::fixed << std::setprecision(4) << diff
            << " close " << close 
            << std::endl 
            ;
    }
    assert(close);   // last cut integral should be close to the avgNumPhotons

    return s2c ; 
}

template NP* QCerenkov::getS2Integral_UpperCut( const double BetaInverse, unsigned nx ) const ; 
template NP* QCerenkov::getS2Integral_UpperCut( const float  BetaInverse, unsigned nx ) const ; 


/**
NP* QCerenkov::getS2Integral_UpperCut
-----------------------------------------

BetaInverse beyond which get no photons could be used to determine 
the bis range, avoiding a large chunk of the output array containing just zeros.

**/

template <typename T>
NP* QCerenkov::getS2Integral_UpperCut( const NP* bis, unsigned nx ) const 
{
    unsigned ni = bis->shape[0] ; 
    const T* bb = bis->cvalues<T>(); 
    unsigned nj = nx ; 

    NP* s2c = NP::Make<T>(ni, nj, 3) ; 
    LOG(info) << "[ creating s2c " << s2c->sstr() ; 

    for(unsigned i=0 ; i < ni ; i++)
    {
        const T BetaInverse = bb[i] ; 
        NP* s2c_one = getS2Integral_UpperCut<T>(BetaInverse, nx ); 
 
        if(s2c_one == nullptr)
        {
            LOG(info) 
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
    LOG(info) << "] creating s2c " << s2c->sstr() ; 
    return s2c ; 
}

template NP* QCerenkov::getS2Integral_UpperCut<double>( const NP* , unsigned ) const ; 
template NP* QCerenkov::getS2Integral_UpperCut<float>(  const NP* , unsigned ) const ; 


/**
QCK QCerenkov::makeICDF_UpperCut
------------------------------------

ny 
    number BetaInverse values "height"
nx
    number of energy domain values "width"

**/

template <typename T>
QCK<T> QCerenkov::makeICDF_UpperCut( unsigned ny, unsigned nx, bool dump) const 
{
    NP* bis = NP::Linspace<T>( 1. , rmx,  ny ) ;  
    NP* avph = getAverageNumberOfPhotons_s2<T>(bis ); 
    NP* s2c = getS2Integral_UpperCut<T>( bis, nx ); 
    NP* s2cn = s2c->copy(); 
    s2cn->divide_by_last<T>(); 

    unsigned nu = 1000 ; 
    unsigned hd_factor = 10 ;
 
    NP* icdf = NP::MakeICDF<T>(s2cn, nu, hd_factor, dump); 
    icdf->set_meta<std::string>("creator", "QCerenkov::makeICDF_UpperCut") ;  
    icdf->set_meta<unsigned>("hd_factor", hd_factor );

    NP* icdf_prop = NP::MakeProperty<T>( icdf, hd_factor ) ;
    icdf_prop->set_meta<std::string>("creator", "QCerenkov::makeICDF_UpperCut") ;  
    icdf_prop->set_meta<unsigned>("hd_factor", hd_factor );


    QCK<T> qck ; 

    qck.rindex = dsrc ; 
    qck.bis = bis ;  
    qck.avph = avph ; 
    qck.s2c = s2c ;
    qck.s2cn = s2cn ; 
    qck.icdf = icdf ; 
    qck.icdf_prop = icdf_prop ; 

    return qck ; 
}
template QCK<double> QCerenkov::makeICDF_UpperCut<double>( unsigned , unsigned, bool ) const ; 
template QCK<float>  QCerenkov::makeICDF_UpperCut<float>(  unsigned , unsigned, bool ) const ; 




template <typename T>
QCK<T> QCerenkov::makeICDF_SplitBin( unsigned ny, unsigned mul, bool dump) const 
{
    NP* bis = NP::Linspace<T>( 1. , rmx,  ny ) ;  
    std::stringstream ss ; 
    ss << "name:makeICDF_SplitBin,mul:" << mul ; 
    bis->meta = ss.str(); 

    NP* avph = getAverageNumberOfPhotons_s2<T>(bis ); 
    NP* s2c = getS2Integral_SplitBin<T>( bis, mul, dump ); 
    NP* s2cn = s2c->copy(); 
    s2cn->divide_by_last<T>(); 

    unsigned nu = 1000 ; 
    unsigned hd_factor = 10 ; 
    NP* icdf = NP::MakeICDF<T>(s2cn, nu, hd_factor, dump); 
    icdf->set_meta<std::string>("creator", "QCerenkov::makeICDF_SplitBin") ;  
    icdf->set_meta<unsigned>("hd_factor", hd_factor );

    NP* icdf_prop = NP::MakeProperty<T>( icdf, hd_factor ) ;
    icdf_prop->set_meta<std::string>("creator", "QCerenkov::makeICDF_UpperCut") ;  
    icdf_prop->set_meta<unsigned>("hd_factor", hd_factor );


    QCK<T> qck ; 

    qck.rindex = dsrc ; 
    qck.bis = bis ;  
    qck.avph = avph ; 
    qck.s2c = s2c ;
    qck.s2cn = s2cn ; 
    qck.icdf = icdf ; 
    qck.icdf_prop = icdf_prop ; 

    return qck ; 
}
template QCK<double> QCerenkov::makeICDF_SplitBin<double>( unsigned , unsigned, bool ) const ; 
template QCK<float>  QCerenkov::makeICDF_SplitBin<float>(  unsigned , unsigned, bool ) const ; 


QTex<float4>* QCerenkov::MakeTex(const NP* icdf) // static
{
    unsigned ndim = icdf->shape.size(); 
    unsigned hd_factor = icdf->get_meta<unsigned>("hd_factor", 0) ; 
    char filterMode = 'P' ;   // 'P' for testing only 

    LOG(LEVEL)
        << "["  
        << " icdf " << icdf->sstr()
        << " ndim " << ndim 
        << " hd_factor " << hd_factor 
        << " filterMode " << filterMode 
        ;

    assert( ndim == 3 && icdf->shape[ndim-1] == 4 ); 

    QTex<float4>* tx = QTexMaker::Make2d_f4(icdf, filterMode ); 
    tx->setHDFactor(hd_factor); 
    tx->uploadMeta(); 

    LOG(LEVEL) << "]" ; 

    return tx ; 
}

void QCerenkov::makeTex(const NP* icdf)
{
    tex = MakeTex(icdf); 
}


extern "C" void QCerenkov_check(dim3 numBlocks, dim3 threadsPerBlock, unsigned width, unsigned height  ); 

template <typename T>
extern void QCerenkov_lookup(dim3 numBlocks, dim3 threadsPerBlock, cudaTextureObject_t texObj, quad4* meta, T* lookup, unsigned num_lookup, unsigned width, unsigned height  ); 


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

