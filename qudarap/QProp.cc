#include <cuda_runtime.h>
#include <sstream>

#include "SStr.hh"
#include "SPath.hh"
#include "scuda.h"
#include "QUDA_CHECK.h"
#include "NP.hh"
#include "QProp.hh"
#include "QU.hh"
#include "qprop.h"
#include "PLOG.hh"


template<typename T>
const plog::Severity QProp<T>::LEVEL = PLOG::EnvLevel("QProp", "INFO"); 

template<typename T>
const char* QProp<T>::DEFAULT_PATH = "$OPTICKS_KEYDIR/GScintillatorLib/LS_ori/RINDEX.npy" ;
//const char* QProp::DEFAULT_PATH = "/tmp/np/test_compound_np_interp.npy" ; 


template<typename T>
const QProp<T>* QProp<T>::INSTANCE = nullptr ; 

template<typename T>
const QProp<T>* QProp<T>::Get(){ return INSTANCE ; }

template<typename T>
qprop<T>* QProp<T>::getDevicePtr() const
{
    return d_prop ; 
}


/**
QProp::Load_Mockup
-------------------

Mockup a real set of multiple properties, by loading 
a single property multiple times and applying scalings. 

The source property is assumed to be provided in double precision 
(ie direct from Geant4 originals) with energies in MeV which are scaled to eV.
Also the properties are narrowed to float when the template type is float.

**/

template<typename T>
const NP* QProp<T>::Load_Mockup(const char* path_ )  // static 
{
    int create_dirs = 0 ;  // 0:nop
    const char* path = SPath::Resolve(path_, create_dirs); 
    LOG(LEVEL) 
        << "path_ " << path_  
        << "path " << path  
        ;

    if( path == nullptr ) return nullptr ; 
    NP* a = NP::Load(path) ; 
    assert( strcmp( a->dtype, "<f8") == 0 ); 
    a->pscale<double>(1e6, 0u);   // energy scale from MeV to eV,   1.55 to 15.5 eV

    NP* b = NP::Load(path); 
    b->pscale<double>(1e6,  0u); 
    b->pscale<double>(1.05, 1u); 

    NP* c = NP::Load(path); 
    c->pscale<double>(1e6,  0u); 
    c->pscale<double>(0.95, 1u); 

    std::vector<const NP*> aa = {a, b, c } ; 
    const NP* com = Combine(aa); 

   LOG(LEVEL) 
        << " com " << ( com ? com->desc() : "-" )
        ;

    return com ; 
}


/**
QProp<T>::Combine
-------------------

Only implemented for float template specialization.

Combination using NP::Combine which pads shorter properties
allowing all to be combined into a single array, with final 
extra column used to record the payload column count.

**/

template<typename T>
const NP* QProp<T>::Combine(const std::vector<const NP*>& aa )   // static
{
    assert(0); 
    return nullptr ;  
}

template<>
const NP* QProp<float>::Combine(const std::vector<const NP*>& aa )   // static
{
    LOG(LEVEL) << " narrowing double to float " ; 
    std::vector<const NP*> nn ; 
    for(unsigned i=0 ; i < aa.size() ; i++)
    {
        const NP* a = aa[i] ; 
        const NP* n = NP::MakeNarrow( a );
        nn.push_back(n); 
    }
    NP* com = NP::Combine(nn) ; 
    return com ;  
}

template<>
const NP* QProp<double>::Combine(const std::vector<const NP*>& aa )   // static
{
    LOG(LEVEL) << " not-narrowing retaining double " ; 
    NP* com = NP::Combine(aa) ;
    return com ;  
}


/**
QProp<T>::QProp
-----------------

Instanciation:

1. loads properties from file
2. creates host qprop<T> instance, and populates it 
   with device pointers and metadata such as dimensions 
3. uploads the host qprop<T> instance to the device, 
   retaining device pointer d_prop

**/

template<typename T>
QProp<T>::QProp(const char* path_)
    :
    path(path_ ? strdup(path_) : DEFAULT_PATH),
    a(Load_Mockup(path)),
    pp(a ? a->cvalues<T>() : nullptr),
    nv(a ? a->num_values() : 0),
    ni(a ? a->shape[0] : 0 ),
    nj(a ? a->shape[1] : 0 ),
    nk(a ? a->shape[2] : 0 ),
    prop(new qprop<T>),
    d_prop(nullptr)
{
    INSTANCE = this ; 
    init(); 
} 

template<typename T>
void QProp<T>::init()
{
    assert( a->uifc == 'f' ); 
    assert( a->ebyte == sizeof(T) );  
    assert( a->shape.size() == 3 ); 

    //dump(); 
    uploadProps(); 
}

template<typename T>
void QProp<T>::uploadProps()
{
    prop->pp = QU::device_alloc<T>(nv) ; 
    prop->height = ni ; 
    prop->width =  nj*nk ; 

    QU::copy_host_to_device<T>( prop->pp, pp, nv ); 

    d_prop = QU::UploadArray<qprop<T>>(prop, 1 );  
}



template<typename T>
void QProp<T>::cleanup()
{
    QUDA_CHECK(cudaFree(prop->pp)); 
    QUDA_CHECK(cudaFree(d_prop)); 
}

template<typename T>
QProp<T>::~QProp()
{
}

template<typename T>
std::string QProp<T>::desc() const 
{
    std::stringstream ss ; 
    ss << "QProp::desc"
       << " path " << ( path ? path : "-" ) 
       << " a " << ( a ? a->desc() : "-" )
       << " nv " << nv
       << " ni " << ni
       << " nj " << nj
       << " nk " << nk
       ;
    return ss.str(); 
}



template<typename T>
void QProp<T>::dump() const 
{
    LOG(info) << desc() ; 
    for(unsigned i=0 ; i < ni ; i++)
    {
        for(unsigned j=0 ; j < nj ; j++)
        {
            for(unsigned k=0 ; k < nk ; k++)
            {
                std::cout 
                    << std::setw(10) << std::fixed << std::setprecision(5) << pp[nk*nj*i+j*nk+k] << " " 
                    ; 
            }
    
            T f = pp[nk*nj*i+j*nk+nk-1] ; 
            unsigned prop_ni  = sview::uint_from<T>(f); 
            std::cout 
                << " prop_ni :" << std::setw(5) << prop_ni 
                << std::endl
                ; 

            assert( prop_ni < nj ) ;
        }
    }
}



// NB this cannot be extern "C" as need C++ name mangling for template types

template <typename T>
extern void QProp_lookup(
    dim3 numBlocks, 
    dim3 threadsPerBlock, 
    qprop<T>* prop, 
    T* lookup, 
    const T* domain, 
    unsigned iprop, 
    unsigned domain_width
); 

template<typename T>
void QProp<T>::lookup( T* lookup, const T* domain,  unsigned lookup_prop, unsigned domain_width ) const 
{
    unsigned num_lookup = lookup_prop*domain_width ; 

    LOG(LEVEL) 
        << "["
        << " lookup_prop " << lookup_prop
        << " domain_width " << domain_width
        << " num_lookup " << num_lookup
        ; 

    T* d_domain = QU::device_alloc<T>(domain_width) ; 
    QU::copy_host_to_device<T>( d_domain, domain, domain_width  ); 

    T* d_lookup = QU::device_alloc<T>(num_lookup) ; 

    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    configureLaunch( numBlocks, threadsPerBlock, domain_width, 1 ); 

    for(unsigned iprop=0 ; iprop < lookup_prop ; iprop++)
    {
        QProp_lookup(numBlocks, threadsPerBlock, d_prop, d_lookup, d_domain, iprop, domain_width );  
    }

    QU::copy_device_to_host_and_free<T>( lookup, d_lookup, num_lookup ); 
     
    LOG(LEVEL) << "]" ; 
}


template<typename T>
void QProp<T>::configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height ) const 
{
    threadsPerBlock.x = 512 ; 
    threadsPerBlock.y = 1 ; 
    threadsPerBlock.z = 1 ; 
 
    numBlocks.x = (width + threadsPerBlock.x - 1) / threadsPerBlock.x ; 
    numBlocks.y = (height + threadsPerBlock.y - 1) / threadsPerBlock.y ;
    numBlocks.z = 1 ; 
}



#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
// quell warning: type attributes ignored after type is already defined [-Wattributes]
template struct QUDARAP_API QProp<float>;
template struct QUDARAP_API QProp<double>;
#pragma GCC diagnostic pop



