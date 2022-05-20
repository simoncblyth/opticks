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
const QProp<T>* QProp<T>::INSTANCE = nullptr ; 

template<typename T>
const QProp<T>* QProp<T>::Get(){ return INSTANCE ; }

template<typename T>
qprop<T>* QProp<T>::getDevicePtr() const
{
    return d_prop ; 
}


/**
QProp<T>::QProp
-----------------

Instanciation:

1. examines input combined array dimensions 
2. creates host qprop<T> instance, and populates it 
   with device pointers and metadata such as dimensions 
3. uploads the host qprop<T> instance to the device, 
   retaining device pointer d_prop

**/

template<typename T>
QProp<T>::QProp(const NP* a_)
    :
    a(a_),
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



