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
    dmin(0.),
    dmax(0.),
    src(dsrc->ebyte == 4 ? dsrc : NP::MakeNarrow(dsrc)),
    tex(nullptr)
{
    INSTANCE = this ; 
    init(); 
}


void QCerenkov::init()
{
    dsrc->pscale<double>(1e6, 0u) ; //  change energy scale from MeV to eV,   1.55 to 15.5 eV
    dsrc->minmax<double>(dmin, dmax, 0u ); 
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
QCerenkov::GetAverageNumberOfPhotons_s
---------------------------------------

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
T QCerenkov::GetAverageNumberOfPhotons_s2(T& emin,  T& emax, const T BetaInverse, const T  charge ) const 
{
    emin = dmax ; // start with inverted range
    emax = dmin ; 
 
    const T* vv = dsrc->cvalues<T>(); 
    unsigned ni = dsrc->shape[0] ; 
    unsigned nj = dsrc->shape[1] ; 
    assert( nj == 2 && ni > 1 ); 

    T s2integral(0.) ;  
    for(unsigned i=0 ; i < ni - 1 ; i++)
    {
        T en_0 = vv[2*(i+0)+0] ; 
        T en_1 = vv[2*(i+1)+0] ; 

        T ri_0 = vv[2*(i+0)+1] ; 
        T ri_1 = vv[2*(i+1)+1] ; 

        s2integral += getS2Integral<T>( emin, emax, BetaInverse, en_0, en_1, ri_0, ri_1 ); 
    }
    const T numPhotons = charge * charge * s2integral ;
    return numPhotons ; 
}


template double QCerenkov::GetAverageNumberOfPhotons_s2<double>(double& emin,  double& emax, const double BetaInverse, const double  charge ) const ;
template float  QCerenkov::GetAverageNumberOfPhotons_s2<float>( float& emin,   float&  emax, const float  BetaInverse, const float   charge ) const ;



const double QCerenkov::FINE_STRUCTURE_OVER_HBARC_EVMM = 36.981 ; 


template<typename T>
T QCerenkov::getS2Integral( T& emin, T& emax, const T BetaInverse, const T en_0, const T en_1 , const T ri_0, const T ri_1 ) const
{
    const T zero(0.) ; 
    const T one(1.) ; 
    const T half(0.5) ; 
    T s2integral(0.) ;  
    T en_cross(0.); 

    T ct_0 = BetaInverse/ri_0 ; 
    T ct_1 = BetaInverse/ri_1 ; 

    T s2_0 = ( one - ct_0 )*( one + ct_0 ); 
    T s2_1 = ( one - ct_1 )*( one + ct_1 ); 

    if( s2_0 <= zero and s2_1 <= zero )   // entire bin disallowed : no contribution 
    {    
        s2integral = zero ; 
    }    
    else if( s2_0 < zero and s2_1 > zero )  // s2 becomes +ve within the bin
    {    
        en_cross = (s2_1*en_0 - s2_0*en_1)/(s2_1 - s2_0) ;   // see ~/np/NP.hh NP::linear_crossings   
        s2integral =  (en_1 - en_cross)*s2_1*half ;  

        emin = std::min<T>( emin, en_cross );  
        emax = std::max<T>( emax, en_1 );
    }    
    else if( s2_0 >= zero and s2_1 >= zero )   // s2 +ve across full bin 
    {    
        s2integral = (en_1 - en_0)*(s2_0 + s2_1)*half ;  

        emin = std::min<T>( emin, en_0 );  
        emax = std::max<T>( emax, en_1 );
    }     
    else if( s2_0 > zero and s2_1 < zero )  // s2 becomes -ve within the bin
    {    
        en_cross = (s2_1*en_0 - s2_0*en_1)/(s2_1 - s2_0) ;   // see ~/np/NP.hh NP::linear_crossings   
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
    const T Rfact = FINE_STRUCTURE_OVER_HBARC_EVMM ;     
    return Rfact*s2integral ; 
}


template double QCerenkov::getS2Integral( double& emin, double& emax, const double BetaInverse, const double en_0, const double en_1 , const double ri_0, const double ri_1 ) const ; 
template float  QCerenkov::getS2Integral( float& emin, float& emax, const float BetaInverse, const float en_0, const float en_1 , const float ri_0, const float ri_1 ) const ; 

/**
QCerenkov::getS2SliverIntegrals
----------------------------------

See ana/rindex.py:s2sliver_integrate

**/

template <typename T>
NP* QCerenkov::getS2SliverIntegrals( T& emin, T& emax, const T BetaInverse, const NP* edom ) const 
{
     emin = dmax ; 
     emax = dmin ; 

     unsigned ni = edom->shape[0] ; 
     const T* ee = edom->cvalues<T>(); 

     NP* s2slv = NP::Make<T>(ni) ; 
     T* ss = s2slv->values<T>(); 

     for(unsigned i=0 ; i < ni-1 ; i++)
     {
         T en_0 = ee[i] ; 
         T en_1 = ee[i+1] ; 

         T ri_0 = dsrc->interp<T>(en_0) ; 
         T ri_1 = dsrc->interp<T>(en_1) ; 

         ss[i+1] = getS2Integral( emin, emax, BetaInverse, en_0, en_1, ri_0, ri_1 );  
     }
     return s2slv ; 
}

template NP* QCerenkov::getS2SliverIntegrals( double& emin, double& emax, const double BetaInverse, const NP* edom ) const ; 
template NP* QCerenkov::getS2SliverIntegrals( float& emin, float& emax, const float BetaInverse, const NP* edom ) const ; 


/**
QCerenkov::getS2SliverIntegrals
---------------------------------

Hmm unlike ana/rindex.py this is using 
the same energy domain for all BetaInverse.

As are immediately going to invert the CDF  
it is better to use an energy range specific to 
each BetaInverse to make the best use of the bins.

**/

template <typename T>
NP* QCerenkov::getS2SliverIntegrals( const NP* bis, const NP* edom ) const 
{
     unsigned ni = bis->shape[0] ; 
     const T* bb = bis->cvalues<T>(); 

     unsigned nj = edom->shape[0] ; 
     const T* ee = edom->cvalues<T>(); 

     NP* s2slv = NP::Make<T>(ni, nj) ; 
     T* ss = s2slv->values<T>(); 

     for(unsigned i=0 ; i < ni ; i++)
     {
         const T BetaInverse = bb[i] ; 
         T emin = dmax ; 
         T emax = dmin ; 

         for(unsigned j=0 ; j < nj-1 ; j++)
         {
             T en_0 = ee[j] ; 
             T en_1 = ee[j+1] ; 

             T ri_0 = dsrc->interp<T>(en_0) ; 
             T ri_1 = dsrc->interp<T>(en_1) ; 

             ss[i*nj+j] = getS2Integral( emin, emax, BetaInverse, en_0, en_1, ri_0, ri_1 );  
         }
         // emin,emax not currently used
     }
     return s2slv ; 
}

template NP* QCerenkov::getS2SliverIntegrals<double>( const NP* bis, const NP* edom ) const ; 
template NP* QCerenkov::getS2SliverIntegrals<float>(  const NP* bis, const NP* edom ) const ; 




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

