#pragma once
/**
stmm.h : Thin/Thick Multi-layer Stack TMM "Transfer Matrix Method" A,R,T calculation 
======================================================================================

Developed in ~/j/Layr/Layr.h, moved into syrap to assist with comparison
with qsim.h propagate_at_boundary 


1. nothing JUNO specific here
2. header-only implementation, installed with j/PMTSim 

See Also
-----------

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

template<typename F> struct ART_ : (3,4) 
    results  

template<typename T, int N> StackSpec :  (4,3) 
    4 sets of complex refractive index and thickness

    * HMM maybe pad to (4,4) if decide to keep ?

template <typename T, int N> struct Stack : (constituents persisted separately) 
    N Layr stack : all calculations in ctor  

    Layr<T> ll[N] ;   
    Layr<T> comp ;  // composite for the N layers 
    ART_<T>  art ; 

    LAYR_METHOD Stack(T wl, T minus_cos_theta, const StackSpec4<T>& ss);

**/

#ifdef WITH_THRUST
#include <thrust/complex.h>
#else
#include <complex>
#include <cmath>
#endif

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cassert>
#include <cstdlib>
#include <vector>
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
    LAYR_METHOD constexpr T pi() { return T(M_PI) ; } 

    template<typename T>
    LAYR_METHOD constexpr T twopi() { return T(2.0*M_PI) ; } 
}


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    #ifdef WITH_THRUST
    template<typename T>
    inline std::ostream& operator<<(std::ostream& os, const thrust::complex<T>& z)  
    {
        os << "(" << std::setw(10) << std::fixed << std::setprecision(4) << z.real() 
           << " " << std::setw(10) << std::fixed << std::setprecision(4) << z.imag() << ")t" ; 
        return os; 
    }
    #else
    template<typename T>
    inline std::ostream& operator<<(std::ostream& os, const std::complex<T>& z)  
    {
        os << "(" << std::setw(10) << std::fixed << std::setprecision(4) << z.real() 
           << " " << std::setw(10) << std::fixed << std::setprecision(4) << z.imag() << ")s" ; 
        return os; 
    }
    #endif  // clarity is more important than brevity 
#endif

template<typename T>
struct Matx
{
#ifdef WITH_THRUST
    thrust::complex<T> M00, M01, M10, M11 ;   
#else
    std::complex<T>    M00, M01, M10, M11 ;       
#endif
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
#ifdef WITH_THRUST
    using thrust::complex ; 
#else
    using std::complex ; 
#endif
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
#ifdef WITH_THRUST 
    thrust::complex<T>  n, st, ct ; 
#else
    std::complex<T>     n, st, ct ;
#endif
    // ---------------------------------------- 1st (4,2)

#ifdef WITH_THRUST 
    thrust::complex<T>  rs, rp, ts, tp ;    
#else
    std::complex<T>     rs, rp, ts, tp ;    
#endif
    // ---------------------------------------- 2nd (4,2)
    Matx<T> S ;                             
    // ---------------------------------------- 3rd (4,2)
    Matx<T> P ;                               
    // ---------------------------------------- 

    LAYR_METHOD void reset(); 
    LAYR_METHOD void load4( const T* vals ); 
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

template<typename F>
struct ART_
{   
    F R_s;     // R_s = a.arts[:,0,0]
    F R_p;     // R_p = a.arts[:,0,1]
    F T_s;     // T_s = a.arts[:,0,2]
    F T_p;     // T_p = a.arts[:,0,3]

    F A_s;     // A_s = a.arts[:,1,0]
    F A_p;     // A_p = a.arts[:,1,1]
    F R;       // R   = a.arts[:,1,2]
    F T;       // T   = a.arts[:,1,3]

    F A;       // A   = a.arts[:,2,0]
    F A_R_T ;  // A_R_T = a.arts[:,2,1] 
    F wl ;     // wl  = a.arts[:,2,2]
    F mct ;    // mct  = a.arts[:,2,3]   

    // persisted into shape (3,4) 
};

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
template<typename T>
inline std::ostream& operator<<(std::ostream& os, const ART_<T>& art )  
{
    os 
        << "ART_" << std::endl 
        << " R_s " << std::fixed << std::setw(10) << std::setprecision(4) << art.R_s 
        << " R_p " << std::fixed << std::setw(10) << std::setprecision(4) << art.R_p 
        << std::endl 
        << " T_s " << std::fixed << std::setw(10) << std::setprecision(4) << art.T_s 
        << " T_p " << std::fixed << std::setw(10) << std::setprecision(4) << art.T_p 
        << std::endl 
        << " A_s " << std::fixed << std::setw(10) << std::setprecision(4) << art.A_s 
        << " A_p " << std::fixed << std::setw(10) << std::setprecision(4) << art.A_p 
        << std::endl 
        << " R   " << std::fixed << std::setw(10) << std::setprecision(4) << art.R   
        << " T   " << std::fixed << std::setw(10) << std::setprecision(4) << art.T   
        << " A   " << std::fixed << std::setw(10) << std::setprecision(4) << art.A  
        << " A_R_T " << std::fixed << std::setw(10) << std::setprecision(4) << art.A_R_T 
        << std::endl 
        << " wl  " << std::fixed << std::setw(10) << std::setprecision(4) << art.wl  << std::endl 
        << " mct " << std::fixed << std::setw(10) << std::setprecision(4) << art.mct << std::endl 
        ;
    return os; 
}
#endif


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
namespace sys
{
    template<typename T>
    inline std::vector<T>* getenvvec(const char* ekey, const char* fallback = nullptr, char delim=',')
    {
        char* _line = getenv(ekey);
        const char* line = _line ? _line : fallback ; 
        if(line == nullptr) return nullptr ; 

        std::stringstream ss; 
        ss.str(line);
        std::string s;

        std::vector<T>* vec = new std::vector<T>() ; 

        while (std::getline(ss, s, delim)) 
        {   
            std::istringstream iss(s);
            T t ; 
            iss >> t ; 
            vec->push_back(t) ; 
        }   
        return vec ; 
    }
}
#endif


template<typename T>
struct LayrSpec
{
    T nr, ni, d ; 
#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    static int EGet(LayrSpec<T>& ls, int idx); 
#endif
};

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
template<typename T>
LAYR_METHOD int LayrSpec<T>::EGet(LayrSpec<T>& ls, int idx)
{
    std::stringstream ss ; 
    ss << "L" << idx ; 
    std::string ekey = ss.str(); 
    std::vector<T>* vls = sys::getenvvec<T>(ekey.c_str()) ; 
    if(vls == nullptr) return 0 ; 
    const T zero(0) ; 
    ls.nr = vls->size() > 0u ? (*vls)[0] : zero ; 
    ls.ni = vls->size() > 1u ? (*vls)[1] : zero ; 
    ls.d  = vls->size() > 2u ? (*vls)[2] : zero ; 
    return 1 ; 
}

template<typename T>
LAYR_METHOD std::ostream& operator<<(std::ostream& os, const LayrSpec<T>& ls )  
{
    os 
        << "LayrSpec<" << ( sizeof(T) == 8 ? "double" : "float" ) << "> "  
        << std::fixed << std::setw(10) << std::setprecision(4) << ls.nr << " "
        << std::fixed << std::setw(10) << std::setprecision(4) << ls.ni << " ; "
        << std::fixed << std::setw(10) << std::setprecision(4) << ls.d  << ")"
        << std::endl 
        ;
    return os ; 
}
#endif


template<typename T, int N>  
struct StackSpec
{
    LayrSpec<T> ls[N] ; 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    LAYR_METHOD void eget() ; 
#endif

}; 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
template<typename T, int N>
LAYR_METHOD void StackSpec<T,N>::eget()  
{
    int count = 0 ; 
    for(int i=0 ; i < N ; i++) count += LayrSpec<T>::EGet(ls[i], i); 
    assert( count == N ) ; 
}

template<typename T, int N>
LAYR_METHOD std::ostream& operator<<(std::ostream& os, const StackSpec<T,N>& ss )  
{
    os 
        << "StackSpec<" 
        << ( sizeof(T) == 8 ? "double" : "float" ) 
        << "," << N
        << ">"  
        << std::endl ;

    for(int i=0 ; i < N ; i++) os << ss.ls[i] ; 
    return os ; 
}

#endif


/**
Stack
-------

**/

template <typename T, int N>
struct Stack
{
    Layr<T> ll[N] ;  
    Layr<T> comp ;  // composite for the N layers 
    ART_<T>  art ; 

    LAYR_METHOD void zero();
    LAYR_METHOD Stack();
    LAYR_METHOD Stack(T wl, T minus_cos_theta, const StackSpec<T,N>& ss);
};

template<typename T, int N>
LAYR_METHOD void Stack<T,N>::zero()
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

SO instead pass in a reference to the object "QPMT.hh/spmt.h" 
that handles the PMT properties, and is responsible for:

1. holding PMT properties (or textures)
2. doing property lookup as function of wavelenth/energy
3. populating Layr<T>* with the results 

HMM: 

* how to test the counterpair pair in a way that can work on both CPU and GPU ?
* dont need to use NP::combined_interpolate_5 GPU compatible lookup code can be used on CPU also 


HMM: more physical to use dot(photon_momentum,outward_surface_normal) 
as "angle" parameter, the dot product is -cos(aoi)

1. -1 at normal incidence against surface_normal, inwards going 
2. +1 at normal incidence with the surface_normal, outwards going  
3.  0 at glancing incidence (90 deg AOI) : potential for math blowouts here 
4. sign of dot product indicates when must flip the stack of parameters
5. angle scan plots can then use aoi 0->180 deg, which is -cos(aoi) -1->1   
   (will there be continuity across the turnaround ?)

**/

template<typename T, int N>
LAYR_METHOD Stack<T,N>::Stack()
{
    zero(); 
}

template<typename T, int N>
LAYR_METHOD Stack<T,N>::Stack(T wl, T minus_cos_theta, const StackSpec<T,N>& ss ) 
{
    // minus_cos_theta, aka dot(mom,normal)
#ifdef WITH_THRUST
    using thrust::complex ; 
    using thrust::norm ; 
    using thrust::conj ; 
    using thrust::exp ; 
    using thrust::sqrt ; 
    using thrust::sin ; 
    using thrust::cos ; 
#else
    using std::complex ; 
    using std::norm ; 
    using std::conj ; 
    using std::exp ; 
    using std::sqrt ; 
    using std::sin ; 
    using std::cos ; 
#endif

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    assert( N >= 2); 
#endif

    const T zero(Const::zero<T>()) ; 
    const T one(Const::one<T>()) ; 
    const T two(Const::two<T>()) ; 
    const T twopi(Const::twopi<T>()) ; 

    for(int i=0 ; i < N ; i++)
    {
        int j = minus_cos_theta < zero ? i : N - 1 - i ;  
        //  minus_cos_theta < zero  : against normal : ordinary stack  : j = i 
        //  minus_cos_theta >= zero : with normal    : backwards stack : j from end 

        ll[j].n.real(ss.ls[i].nr) ; 
        ll[j].n.imag(ss.ls[i].ni) ; 
        ll[j].d = ss.ls[i].d ; 
    }

    // ll[0]   is "top"     : start layer : incident
    // ll[N-1] is "bottom"  : end   layer : transmitted


    art.wl = wl ; 
    art.mct = minus_cos_theta ; 

    const complex<T> zOne(one,zero); 
    const complex<T> zI(zero,one); 
    const complex<T> mct(minus_cos_theta);  // simpler for everything to be complex

    // Snell : set st,ct of all layers (depending on indices(hence wl) and incident angle) 
    Layr<T>& l0 = ll[0] ; 
    l0.ct = minus_cos_theta < zero  ? -mct : mct ; 
    //
    //  flip picks +ve ct that constrains the angle to first quadrant 
    //  this works as : cos(pi-theta) = -cos(theta)
    //  without flip, the ART values are non-physical : always outside 0->1 for mct > 0 angle > 90  
    //

    l0.st = sqrt( zOne - mct*mct ) ;  

    for(int idx=1 ; idx < N ; idx++)  // for N=2 idx only 1, sets only ll[1] 
    {
        Layr<T>& l = ll[idx] ; 
        l.st = l0.n * l0.st / l.n  ; 
        l.ct = sqrt( zOne - l.st*l.st );
    }     

    // Fresnel : set rs/rp/ts/tp for N-1 interfaces between N layers
    // (depending on indices(hence wl) and layer ct) 
    // HMM: last layer unset, perhaps zero it ?
    for(int idx=0 ; idx < N-1 ; idx++)
    {
        // cf OpticalSystem::Calculate_rt  
        // https://en.wikipedia.org/wiki/Fresnel_equations
        Layr<T>& i = ll[idx] ; 
        const Layr<T>& j = ll[idx+1] ;  

        i.rs = (i.n*i.ct - j.n*j.ct)/(i.n*i.ct+j.n*j.ct) ;  // r_s eoe[12] see g4op-eoe
        i.rp = (j.n*i.ct - i.n*j.ct)/(j.n*i.ct+i.n*j.ct) ;  // r_p eoe[7]
        i.ts = (two*i.n*i.ct)/(i.n*i.ct + j.n*j.ct) ;       // t_s eoe[9]
        i.tp = (two*i.n*i.ct)/(j.n*i.ct + i.n*j.ct) ;       // t_p eoe[8]
    }

    // populate transfer matrices for both thick and thin layers  
    // for N=2 only one interface

    ll[0].reset();    // ll[0].S ll[0].P matrices set to identity 

    for(int idx=1 ; idx < N ; idx++)
    {
        const Layr<T>& i = ll[idx-1] ;            
        Layr<T>& j       = ll[idx] ;          
        // looking at (i,j) pairs of layers 

        complex<T> tmp_s = one/i.ts ; 
        complex<T> tmp_p = one/i.tp ;   
        // at glancing incidence ts, tp approach zero : blowing up tmp_s tmp_p
        // which causes the S and P matrices to blow up yielding infinities at mct zero
        //
        // thick layers indicated with d = 0. 
        // thin layers have thickness presumably comparable to art.wl (WITH SAME LENGTH UNIT: nm)
        complex<T> delta         = j.d == zero ? zero : twopi*j.n*j.d*j.ct/art.wl ; 
        complex<T> exp_neg_delta = j.d == zero ? one  : exp(-zI*delta) ; 
        complex<T> exp_pos_delta = j.d == zero ? one  : exp( zI*delta) ; 

        j.S.M00 = tmp_s*exp_neg_delta      ; j.S.M01 = tmp_s*i.rs*exp_pos_delta ; 
        j.S.M10 = tmp_s*i.rs*exp_neg_delta ; j.S.M11 =      tmp_s*exp_pos_delta ; 

        j.P.M00 = tmp_p*exp_neg_delta      ; j.P.M01 = tmp_p*i.rp*exp_pos_delta ; 
        j.P.M10 = tmp_p*i.rp*exp_neg_delta ; j.P.M11 =      tmp_p*exp_pos_delta ; 

        // NB: for thin layers the transfer matrices combine interface and propagation between them 
    }

    /*
    For d=0 (thick layer) 


    S     |  1/ts    rs/ts |    
          |                |
          |  rs/ts   1/ts  |     


    P     |  1/tp    rp/tp |    
          |                |
          |  rp/tp   1/tp  |     

    */



    // product of the transfer matrices
    comp.d = zero ; 
    comp.st = zero ; 
    comp.ct = zero ; 
    comp.S.reset(); 
    comp.P.reset(); 

    for(int idx=0 ; idx < N ; idx++) // TODO: start from idx=1 as ll[0].S ll[0].P always identity
    {
        const Layr<T>& l = ll[idx] ; 
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

    Layr<T>& t = ll[0] ; 
    Layr<T>& b = ll[N-1] ; 

    // getting from amplitude to power relations for relectance R (same material and angle) and tranmittance T
    //  https://www.brown.edu/research/labs/mittleman/sites/brown.edu.research.labs.mittleman/files/uploads/lecture13_0.pdf

    complex<T> _T_s = (b.n*b.ct)/(t.n*t.ct)*norm(comp.ts) ;  
    complex<T> _T_p = (conj(b.n)*b.ct)/(conj(t.n)*t.ct)*norm(comp.tp) ; 
    // _T_p top and bot layers usually with real index ? so the conj above just noise ?

    art.R_s = norm(comp.rs) ; 
    art.R_p = norm(comp.rp) ; 
    art.T_s = _T_s.real() ; 
    art.T_p = _T_p.real() ; 

    // absorption factor by subtracting reflection and transmission
    art.A_s = one-art.R_s-art.T_s;
    art.A_p = one-art.R_p-art.T_p;

    // average of S and P 
    art.R   = (art.R_s+art.R_p)/two ;
    art.T   = (art.T_s+art.T_p)/two ;
    art.A   = (art.A_s+art.A_p)/two ;
    art.A_R_T = art.A + art.R + art.T ;  
}

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
template <typename T, int N>
inline std::ostream& operator<<(std::ostream& os, const Stack<T,N>& stk )  
{
    os << "Stack"
       << "<" 
       << ( sizeof(T) == 8 ? "double" : "float" )
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

