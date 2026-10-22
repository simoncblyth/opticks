#pragma once
/**
S4MultiLayrStack.h : Thin/Thick Multi-layer Stack TMM "Transfer Matrix Method" A,R,T calculation 
=====================================================================================================

This ~/opticks/sysrap/S4MultiLayrStack.h header is adapted from ~/customgeant4/C4MultiLayrStack.h
with primary change being the use of scomplex_select.h to pick between three complex impl:

1. ~/opticks/sysrap/scomplex.h

   * lite impl - needs testing and comparison with below 

2. <thrust/complex.h>

   * CUDA heavy system of headers - better to avoid in device code (especially optix kernels) if possible

3. <complex>

   * CPU side std::complex


Following notes on ~/customgeant4/C4MultiLayrStack.h
-----------------------------------------------------

This was developed and tested as Layr.h at https://github.com/simoncblyth/j/blob/main/Layr/Layr.h

1. nothing JUNO specific here
2. header-only implementation


See Also tests and notes within https://github.com/simoncblyth/j/blob/main/Layr
----------------------------------------------------------------------------------

Layr.rst 
    notes/refs on TMM theory and CPU+GPU implementation 

LayrTest.{h,cc,cu,py,sh} 
    build, run cpu+gpu scans, plotting, comparisons float/double cpu/gpu std/thrust 

JPMT.h
    JUNO specific collection of PMT rindex and thickness into arrays 


Contents of Layr.h : (persisted array shapes)
----------------------------------------------

namespace Const 
    templated constexpr functions: zero, one, two, pi, twopi

template<typename T> struct Matx : (4,2)
    2x2 complex matrix  

template<typename T> struct Layr : (4,4,2)
    d:thickness, complex refractive index, angular and Fresnel coeff,  S+P Matx

    * d = zero : indicates thick (incoherent) layer 

template<typename F> struct ART_ : (4,4) 
    results  

template<typename T, int N> StackSpec :  (4,4) 
    4 sets of complex refractive index and thickness

template <typename T, int N> struct Stack : (constituents persisted separately) 
    N Layr stack : all calculations in ctor  

    Layr<T> ll[N] ;    (4,4,4,2)  (32,4)    when N=4                            
    Layr<T> comp ;     (1,4,4,2)  (8, 4)    composite for the N layers    (
    ART_<T>  art ;     (4,4)      (4, 4)
                                 --------
                                  (44,4)

    LAYR_METHOD Stack(T wl, T minus_cos_theta, const StackSpec4<T>& ss);

**/


#include "scomplex_select.h"

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cassert>
#include <cstdlib>
#include <vector>
#include <array>
#endif

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define LAYR_METHOD __host__ __device__ __forceinline__
#else
#    define LAYR_METHOD inline 
#endif


/**
Const
-------

The point of these Const functions is to plant the desired type of constant
into assembly code with no runtime transients of another type. Important for
avoiding expensive unintentional doubles in GPU code. The constexpr means that
the conversions and calculations happen at compile time, NOT runtime. 

**/

namespace Const
{
    template<typename T>  
    LAYR_METHOD constexpr T zero(){ return T(0.0) ; } 
 
    template<typename T>
    LAYR_METHOD constexpr T one() { return T(1.0) ; } 

    template<typename T>
    LAYR_METHOD constexpr T two() { return T(2.0) ; } 
    
    template<typename T>
    LAYR_METHOD constexpr T pi_() { return T(M_PI) ; } 

    template<typename T>
    LAYR_METHOD constexpr T twopi_() { return T(2.0*M_PI) ; } 
}


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    template<typename T>
    inline std::ostream& operator<<(std::ostream& os, const opticks::complex<T>& z)  
    {
        os << "(" << std::setw(10) << std::fixed << std::setprecision(4) << z.real() 
           << " " << std::setw(10) << std::fixed << std::setprecision(4) << z.imag() << ")t" ; 
        return os; 
    }
#endif

template<typename T>
struct Matx
{
    opticks::complex<T> M00, M01, M10, M11 ;   
    LAYR_METHOD void reset();             
    LAYR_METHOD void dot(const Matx<T>& other); 
};

template<typename T>
LAYR_METHOD void Matx<T>::reset()
{
    M00.real(1) ; M00.imag(0) ; // conversion from int 
    M01.real(0) ; M01.imag(0) ; 
    M10.real(0) ; M10.imag(0) ; 
    M11.real(1) ; M11.imag(0) ; 
}
/**

      | T00  T01  |  |  M00   M01 | 
      |           |  |            | 
      | T10  T11  |  |  M10   M11 | 

**/
template<typename T>
LAYR_METHOD void Matx<T>::dot(const Matx<T>& other)
{
    using opticks::complex ; 
    complex<T> T00(M00) ; 
    complex<T> T01(M01) ; 
    complex<T> T10(M10) ; 
    complex<T> T11(M11) ; 

    M00 = T00*other.M00 + T01*other.M10;
    M01 = T00*other.M01 + T01*other.M11;
    M10 = T10*other.M00 + T11*other.M10;
    M11 = T10*other.M01 + T11*other.M11;
}

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
template<typename T>
inline std::ostream& operator<<(std::ostream& os, const Matx<T>& m)  
{
    os 
       << "| " << m.M00 << " " << m.M01 << " |" << std::endl 
       << "| " << m.M10 << " " << m.M11 << " |" << std::endl  
       ;
    return os; 
}
#endif


/**
Layr : (4,4,2) 
-----------------

The comp layers do not have the 0th (4,2) filled::

   assert np.all( e.f.comps[:,0] == 0 ) 

**/

template<typename T>
struct Layr
{
    // ---------------------------------------- 0th (4,2)
    T  d ;
    T  pad=0 ;
    opticks::complex<T>  n, st, ct ; 
    // ---------------------------------------- 1st (4,2)
    opticks::complex<T>  rs, rp, ts, tp ;    
    // ---------------------------------------- 2nd (4,2)
    Matx<T> S ;                             
    // ---------------------------------------- 3rd (4,2)
    Matx<T> P ;                               
    // ---------------------------------------- 

    LAYR_METHOD void reset(); 
    LAYR_METHOD void load4( const T* vals ); 
    LAYR_METHOD const T* cdata() const { return &d ; }
};

template<typename T>
LAYR_METHOD void Layr<T>::reset()
{
    S.reset(); 
    P.reset(); 
}

template<typename T>
LAYR_METHOD void Layr<T>::load4(const T* vals)
{
    d   = vals[0] ; 
    pad = vals[1] ; 
    n.real(vals[2]) ; 
    n.imag(vals[3]) ; 
}



#if defined(__CUDACC__) || defined(__CUDABE__)
#else
template<typename T>
inline std::ostream& operator<<(std::ostream& os, const Layr<T>& l)  
{
    os 
       << "Layr"
       << std::endl 
       << std::setw(4) << "n:" << l.n  
       << std::setw(4) << "d:" << std::fixed << std::setw(10) << std::setprecision(4) << l.d  
       << std::endl 
       << std::setw(4) << "st:" << l.st 
       << std::setw(4) << "ct:" << l.ct
       << std::endl 
       << std::setw(4) << "rs:" << l.rs 
       << std::setw(4) << "rp:" << l.rp
       << std::endl 
       << std::setw(4) << "ts:" << l.ts 
       << std::setw(4) << "tp:" << l.tp
       << std::endl 
       << "S" 
       << std::endl 
       << l.S 
       << std::endl 
       << "P"
       << std::endl 
       << l.P
       << std::endl 
       ;
    return os; 
}
#endif


/**
ART_ : rationalized layout  
----------------------------------------

+---+--------+--------+--------+--------+
|   |  x     |  y     |  z     |  w     |
+===+========+========+========+========+
| 0 |  A_s   |  A_p   |  A_av  |  A     |
+---+--------+--------+--------+--------+
| 1 |  R_s   |  R_p   |  R_av  |  R     |
+---+--------+--------+--------+--------+
| 2 |  T_s   |  T_p   |  T_av  |  T     |
+---+--------+--------+--------+--------+
| 3 |  SF    |  wl    |  dpcmn |  mct   |
+---+--------+--------+--------+--------+

Checking A+R+T is very close to 1.::

    art = f.art.squeeze() 
    art_sum = np.sum(art[:,:3], axis=1 )   
    one_dev =  np.abs( art_sum - 1. ).max() 
    assert one_dev < 1e-6 

**/

template<typename F>
struct ART_
{
    F A_s;     // A_s  = a.arts[:,0,0]
    F A_p;     // A_p  = a.arts[:,0,1]
    F A_av;    // A_av = a.arts[:,0,2]
    F A ;      // A    = a.arts[:,0,3]

    F R_s;     // R_s = a.arts[:,1,0]
    F R_p;     // R_p = a.arts[:,1,1]
    F R_av;    // R_av= a.arts[:,1,2]
    F R ;      // R   = a.arts[:,1,3]

    F T_s;     // T_s  = a.arts[:,2,0]
    F T_p;     // T_p  = a.arts[:,2,1]
    F T_av;    // T_av = a.arts[:,2,2]
    F T ;      // T    = a.arts[:,2,3]

    F SF ;      // SF       = a.arts[:,3,0]     S_pol vs P_pol power fraction 
    F wl ;      // wl       = a.arts[:,3,1]
    F dpcmn ;   // dpcmn    = a.arts[:,3,2]    dot_pol_cross_mom_nrm,   formerly ART_av (which is always 1.)
    F mct ;     // mct      = a.arts[:,3,3]   

    LAYR_METHOD const F* cdata() const { return &A_s ; }
    // persisted into shape (4,4) 
};



#if defined(__CUDACC__) || defined(__CUDABE__)
#else
template<typename F>
inline std::ostream& operator<<(std::ostream& os, const ART_<F>& art )  
{
    os 
        << "ART_" 
        << std::endl 
        << "  A_s  " << std::fixed << std::setw(10) << std::setprecision(4) << art.A_s 
        << "  A_p  " << std::fixed << std::setw(10) << std::setprecision(4) << art.A_p 
        << "  A_av " << std::fixed << std::setw(10) << std::setprecision(4) << art.A_av  
        << "  A    " << std::fixed << std::setw(10) << std::setprecision(4) << art.A  
        << std::endl 
        << "  R_s  " << std::fixed << std::setw(10) << std::setprecision(4) << art.R_s 
        << "  R_p  " << std::fixed << std::setw(10) << std::setprecision(4) << art.R_p 
        << "  R_av " << std::fixed << std::setw(10) << std::setprecision(4) << art.R_av   
        << "  R    " << std::fixed << std::setw(10) << std::setprecision(4) << art.R
        << std::endl 
        << "  T_s  " << std::fixed << std::setw(10) << std::setprecision(4) << art.T_s 
        << "  T_p  " << std::fixed << std::setw(10) << std::setprecision(4) << art.T_p 
        << "  T_av " << std::fixed << std::setw(10) << std::setprecision(4) << art.T_av   
        << "  T    " << std::fixed << std::setw(10) << std::setprecision(4) << art.T   
        << std::endl 
        << " SUM_s " << std::fixed << std::setw(10) << std::setprecision(4) << art.A_s + art.R_s + art.T_s
        << " SUM_p " << std::fixed << std::setw(10) << std::setprecision(4) << art.A_p + art.R_p + art.T_p
        << " SUM_a " << std::fixed << std::setw(10) << std::setprecision(4) << art.A_av + art.R_av + art.T_av
        << " SUM_  " << std::fixed << std::setw(10) << std::setprecision(4) << art.A    + art.R    + art.T
        << std::endl 
        << " SF    " << std::fixed << std::setw(10) << std::setprecision(4) << art.SF 
        << " wl    " << std::fixed << std::setw(10) << std::setprecision(4) << art.wl 
        << " dpcmn " << std::fixed << std::setw(10) << std::setprecision(4) << art.dpcmn 
        << " mct   " << std::fixed << std::setw(10) << std::setprecision(4) << art.mct
        << std::endl 
        ;
    return os; 
}

template<typename F>
LAYR_METHOD std::ostream& operator<<(std::ostream& os, const std::array<F,16>& aa )
{
    // curiously fails to template match with generic int N  template param
    os << std::endl ; 
    for(int i=0 ; i < 16 ; i++) os 
        << ( i % 4 == 0 ? '\n' : ' ' ) 
        << std::setw(10) << std::fixed << std::setprecision(4) << aa[i] 
        ;   
    os << std::endl ; 
    return os ; 
}

#endif


/**
Stack
-------

**/

template <typename F, int N>
struct Stack
{
    Layr<F> ll[N] ; //  (4,4,4,2)     when N=4  
    Layr<F> comp ;  //  (1,4,4,2)     composite for the N layers 
    ART_<F>  art ;  //  (4,4) 

    LAYR_METHOD void zero();
    LAYR_METHOD Stack();
    LAYR_METHOD Stack(    F wl, F minus_cos_theta, F dot_pol_cross_mom_nrm, const F* ss, unsigned num_ss );
    LAYR_METHOD void calc(F wl, F minus_cos_theta, F dot_pol_cross_mom_nrm, const F* ss, unsigned num_ss );

    LAYR_METHOD const F* cdata() const { return ll[0].cdata() ; }

};

template<typename F, int N>
LAYR_METHOD void Stack<F,N>::zero()
{
    art = {} ; 
    comp = {} ; 
    for(int i=0 ; i < N ; i++) ll[i] = {} ; 
}


/**
Stack::Stack
---------------

Caution that StackSpec contains refractive indices that depend on wavelength, 
so the wavelength dependency enters twice. 

*minus_cos_theta* which is "dot(photon_momentum,outward_surface_normal)" 
is used as a more physical "angle" parameter, it corresponds to -cos(aoi)

1. minus_cos_theta = -1.f at normal incidence against surface_normal, inwards going 
2. minus_cos_theta = +1.f at normal incidence with the surface_normal, outwards going  
3. minus_cos_theta =  0.f at glancing incidence (90 deg AOI) : potential for math blowouts here 
4. sign of dot product indicates when must flip the stack of parameters
5. angle scan plots can then use aoi 0->180 deg, which is -cos(aoi) -1->1, there 
   should be continuity across the flip otherwise something very wrong

**/

template<typename F, int N>
LAYR_METHOD Stack<F,N>::Stack()
{
    zero(); 
}

/**
Stack::Stack ctor : does TMM A,R,T calc
------------------------------------------

1. Layr populate : ll[0..N-1].n ll[0..N-1].d by copying in from StackSpec

   * HMM: could copying from stackspec be avoided ? Not easily as copying into complex rindex  

2. l0.ct l0.st from minus_cos_theta 

**/

template<typename F, int N>
LAYR_METHOD Stack<F,N>::Stack(F wl, F minus_cos_theta, F dot_pol_cross_mom_nrm, const F* ss, unsigned num_ss ) 
{
    calc(wl, minus_cos_theta, dot_pol_cross_mom_nrm, ss, num_ss ); 
}

/**
Stack::calc
--------------

1. loading stack layers "ll" from ss "stackspec"

* minus_cos_theta < zero  : against normal : ordinary stack  : j = i 
* minus_cos_theta >= zero : with normal    : backwards stack : j from end 

NB stack flipping based on sign of minus_cos_theta ensures consistent layer ordering
     
* ll[0]   is always "top"     : start layer : incident
* ll[N-1] is always "bottom"  : end   layer : transmitted
    

Usage for example from sysrap/SPMT.h SPMT::get_ARTE

* HMM: for mct < 0.f (ingoing photons with efficiency potential) 
  are calling twice, first with (mct,dpcmn) (-1.f,0.f) and then actual (mct,dpcmn) 

  * maybe could skip the stack layer loading on second call ? 
  * Benefit probably so small that overheads would swamp them. 


assert does nothing when NDEBUG macro is defined, so use that to avoid warning
--------------------------------------------------------------------------------

* https://en.cppreference.com/w/cpp/error/assert

**/


template<typename F, int N>
#ifdef NDEBUG
void LAYR_METHOD Stack<F,N>::calc(F wl, F minus_cos_theta, F dot_pol_cross_mom_nrm, const F* ss, unsigned        )
#else
void LAYR_METHOD Stack<F,N>::calc(F wl, F minus_cos_theta, F dot_pol_cross_mom_nrm, const F* ss, unsigned num_ss )
#endif
{
    using opticks::complex ; 
    using opticks::norm ; 
    using opticks::conj ; 
    using opticks::exp ; 
    using opticks::sqrt ; 
    using opticks::sin ; 
    using opticks::cos ; 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    assert( N >= 2); 
    assert( num_ss == N*4 ); 
#endif

    const F zero(Const::zero<F>()) ; 
    const F one(Const::one<F>()) ; 
    const F two(Const::two<F>()) ; 
    const F twopi_(Const::twopi_<F>()) ; 

    for(int i=0 ; i < N ; i++)
    {
        int j = minus_cos_theta < zero ? i : N - 1 - i ;  

        ll[j].n.real(ss[i*4+0]) ; 
        ll[j].n.imag(ss[i*4+1]) ; 
        ll[j].d = ss[i*4+2] ;  
    }

    art.wl = wl ; 
    art.mct = minus_cos_theta ; 
    art.dpcmn = dot_pol_cross_mom_nrm ; 

    const complex<F> zOne(one,zero); 
    const complex<F> zI(zero,one); 
    const complex<F> mct(minus_cos_theta);  // simpler for everything to be complex

    // Snell : set st,ct of all layers (depending on indices(hence wl) and incident angle) 
    Layr<F>& l0 = ll[0] ; 
    l0.ct = minus_cos_theta < zero  ? -mct : mct ; 
    //
    //  flip picks +ve ct that constrains the angle to first quadrant 
    //  this works as : cos(pi-theta) = -cos(theta)
    //  without flip, the ART values are non-physical : always outside 0->1 for mct > 0 angle > 90  
    //

    const complex<F> stst = zOne - mct*mct ; 
    l0.st = sqrt(stst) ; 
      
    /*
    BUG FIX July 2023: 
        using norm(stst) to give sin_theta*sin_theta is incorrect
        because stst is already squared and the norm squares it again 
        Found that bug, because it led to SF of 2. which is nonsensical (SF must be from 0. to 1. ) 
        and also gave a polarization specific A of greater than 1. 
        Test that revealed the bug : qudarap/tests/QSim_MockTest.sh
          
    BUG: const F E_s2 = norm(stst)   > zero ? (dot_pol_cross_mom_nrm*dot_pol_cross_mom_nrm)/norm(stst) : zero ;
    OK : const F E_s2 = l0.st.real() > zero ? (dot_pol_cross_mom_nrm*dot_pol_cross_mom_nrm)/(l0.st.real()*l0.st.real()) : zero ;
    */
    const F E_s2 = stst.real() > zero ? (dot_pol_cross_mom_nrm*dot_pol_cross_mom_nrm)/stst.real() : zero ;

    // E_s2 is the S_polarization vs P_polarization power fraction  
    // E_s2 matches E1_perp*E1_perp see sysrap/tests/stmm_vs_sboundary_test.cc 

#ifdef MOCK_CURAND_DEBUG
    printf("//stack.calc dot_pol_cross_mom_nrm %7.3f norm(stst) %7.3f l0.st (%7.3f,%7.3f) E_s2 %7.3f \n", 
            dot_pol_cross_mom_nrm, norm(stst), l0.st.real(), l0.st.imag(), E_s2 ); 
#endif



    for(int idx=1 ; idx < N ; idx++)  // for N=2 idx only 1, sets only ll[1] 
    {
        Layr<F>& l = ll[idx] ; 
        l.st = l0.n * l0.st / l.n  ; 
        l.ct = sqrt( zOne - l.st*l.st );
    }     

    // Fresnel : set rs/rp/ts/tp for N-1 interfaces between N layers
    // (depending on indices(hence wl) and layer ct) 
    // HMM: last layer unset, perhaps zero it ?
    // cf OpticalSystem::Calculate_rt  
    // https://en.wikipedia.org/wiki/Fresnel_equations

    for(int idx=0 ; idx < N-1 ; idx++)
    {
        Layr<F>& i = ll[idx] ; 
        const Layr<F>& j = ll[idx+1] ;  

        i.rs = (i.n*i.ct - j.n*j.ct)/(i.n*i.ct+j.n*j.ct) ;  // r_s eoe[12] see g4op-eoe
        i.rp = (j.n*i.ct - i.n*j.ct)/(j.n*i.ct+i.n*j.ct) ;  // r_p eoe[7]
        i.ts = (two*i.n*i.ct)/(i.n*i.ct + j.n*j.ct) ;       // t_s eoe[9]
        i.tp = (two*i.n*i.ct)/(j.n*i.ct + i.n*j.ct) ;       // t_p eoe[8]
    }


    /**
    Populate Layer S,P transfer matrices, by looping over consequtive layer pairs (i,j)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Note that for N=2 there is only one interface, for N=4 there are 3 interfaces. 

    At glancing incidence ts, tp approach zero : blowing up tmp_s tmp_p
    which causes the S and P matrices to blow up yielding infinities at mct zero
    
    * thick layers indicated with d = 0. 
    * thin layers have thickness presumably comparable to art.wl (WITH SAME LENGTH UNIT: nm)

    Note that these are the TMM transfer matrices that combine the 
    interfaces and propagation between them : exp_neg_delta, exp_pos_delta).  

    For d=0 (thick layer) the exponents collape to one giving:


    S     |  1/ts    rs/ts |    
          |                |
          |  rs/ts   1/ts  |     


    P     |  1/tp    rp/tp |    
          |                |
          |  rp/tp   1/tp  |     

    **/

    ll[0].reset();    // ll[0].S ll[0].P matrices set to identity 

    for(int idx=1 ; idx < N ; idx++)
    {
        const Layr<F>& i = ll[idx-1] ;            
        Layr<F>& j       = ll[idx] ;          

        complex<F> tmp_s = one/i.ts ; 
        complex<F> tmp_p = one/i.tp ;   

        complex<F> delta         = j.d == zero ? zero : twopi_*j.n*j.d*j.ct/art.wl ; 
        complex<F> exp_neg_delta = j.d == zero ? one  : exp(-zI*delta) ; 
        complex<F> exp_pos_delta = j.d == zero ? one  : exp( zI*delta) ; 

        j.S.M00 = tmp_s*exp_neg_delta      ; j.S.M01 = tmp_s*i.rs*exp_pos_delta ; 
        j.S.M10 = tmp_s*i.rs*exp_neg_delta ; j.S.M11 =      tmp_s*exp_pos_delta ; 

        j.P.M00 = tmp_p*exp_neg_delta      ; j.P.M01 = tmp_p*i.rp*exp_pos_delta ; 
        j.P.M10 = tmp_p*i.rp*exp_neg_delta ; j.P.M11 =      tmp_p*exp_pos_delta ; 

    }


    /**
    Compute composite "effective" layer
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    By product of the transfer matrices, then fishing out the parts. 

    **/

    // product of the transfer matrices
    comp.d = zero ; 
    comp.st = zero ; 
    comp.ct = zero ; 
    comp.S.reset(); 
    comp.P.reset(); 

    for(int idx=0 ; idx < N ; idx++) // TODO: start from idx=1 as ll[0].S ll[0].P always identity
    {
        const Layr<F>& l = ll[idx] ; 
        comp.S.dot(l.S) ; 
        comp.P.dot(l.P) ; 
    }
    // at glancing incidence the matrix from 
    // one of the layers has infinities, which 
    // yields nan in the matrix product 
    // and yields nan for all the below Fresnel coeffs 
    //
    // extract amplitude factors from the composite matrix
    comp.rs = comp.S.M10/comp.S.M00 ; 
    comp.rp = comp.P.M10/comp.P.M00 ; 
    comp.ts = one/comp.S.M00 ; 
    comp.tp = one/comp.P.M00 ; 

    Layr<F>& t = ll[0] ; 
    Layr<F>& b = ll[N-1] ; 

    // getting from amplitude to power relations for relectance R (same material and angle) and tranmittance T
    //  https://www.brown.edu/research/labs/mittleman/sites/brown.edu.research.labs.mittleman/files/uploads/lecture13_0.pdf

    complex<F> _T_s = (b.n*b.ct)/(t.n*t.ct)*norm(comp.ts) ;  
    complex<F> _T_p = (conj(b.n)*b.ct)/(conj(t.n)*t.ct)*norm(comp.tp) ; 
    // _T_p top and bot layers usually with real index ? so the conj above just noise ?

    art.R_s = norm(comp.rs) ; 
    art.R_p = norm(comp.rp) ; 
    art.T_s = _T_s.real() ; 
    art.T_p = _T_p.real() ; 
    art.A_s = one-art.R_s-art.T_s;  // absorption by subtracting reflection and transmission from one
    art.A_p = one-art.R_p-art.T_p;

    const F& SF = E_s2 ;  // combine (_s,_p) appropriate for (minus_cos_theta, dot_pol_cross_mom_nrm )
    art.A = SF*art.A_s + (one-SF)*art.A_p ; 
    art.R = SF*art.R_s + (one-SF)*art.R_p ; 
    art.T = SF*art.T_s + (one-SF)*art.T_p ;  
    // an earlier incarnation of art.T matched with Geant4 TransCoeff see sysrap/tests/stmm_vs_sboundary_test.cc
    art.SF = SF ;

#ifdef MOCK_CURAND_DEBUG
    printf("//Stack::calc SF %7.3f \n", art.SF ); 
#endif

    art.A_av   = (art.A_s+art.A_p)/two ;
    art.R_av   = (art.R_s+art.R_p)/two ;
    art.T_av   = (art.T_s+art.T_p)/two ;

    /**
    HMM : the _av are pointless, only used for normal incidence.
    At normal incidence S/P distinction is meaningless, causing::

        A_s == A_p == A_av == A  
        R_s == R_p == R_av == R  
        T_s == T_p == T_av == T  
    **/

}

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
template <typename F, int N>
inline std::ostream& operator<<(std::ostream& os, const Stack<F,N>& stk )  
{
    os << "Stack"
       << "<" 
       << ( sizeof(F) == 8 ? "double" : "float" )
       << ","
       << N 
       << ">" 
       << std::endl
       ; 
    for(int idx=0 ; idx < N ; idx++) os << "idx " << idx << std::endl << stk.ll[idx] ; 
    os << "comp" 
       << std::endl 
       << stk.comp 
       << std::endl 
       << stk.art 
       << std::endl 
       ; 
    return os; 
}
#endif

