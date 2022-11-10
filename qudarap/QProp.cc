#include <cuda_runtime.h>
#include "SLOG.hh"
#include "NP.hh"

#include "QUDA_CHECK.h"
#include "QU.hh"

#include "QProp.hh"
#include "qprop.h"


template<typename T>
const plog::Severity QProp<T>::LEVEL = SLOG::EnvLevel("QProp", "DEBUG"); 

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
QProp::Make3D  : NOW REMOVED : SHOULD ADJUST SHAPE BEFORE USING CTOR 
-----------------------------------------------------------------------

* QProp requires 1+2D (num_prop, num_energy, 2 )
* BUT: real world arrays such as those from JPMT.h often have more dimensions 3+2D::

      (num_pmtcat, num_layer, num_prop, num_energy, 2)   

* to avoid code duplication or complicated template handling 
  of different shapes, this takes the approach of using NP::change_shape 
  to scrunch up the higher dimensions yielding::

      (num_pmtcat*num_layer*num_prop, num_energy, 2 )

* as there will usually be other dimensional needs in future, its 
  more sustainable to standardize to keep things simple at the 
  expense of requiring a simple calc to get access the 
  scrunched "iprop" eg:: 

    int iprop = pmtcat*NUM_LAYER*NUM_PROP + layer*NUM_PROP + prop_index ;     

HMM can do equivalent of NP::combined_interp_5 

template<typename T>
QProp<T>* QProp<T>::Make3D(const NP* a)
{
    NP* b = a->copy() ; 
    b->change_shape_to_3D(); 
    return new QProp<T>(b) ; 
}
**/


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
    init(); 
} 

template<typename T>
void QProp<T>::init()
{
    INSTANCE = this ; 
    assert( a->uifc == 'f' ); 
    assert( a->shape.size() == 3 ); 
    assert( nv == ni*nj*nk ) ; 

    bool type_consistent = a->ebyte == sizeof(T) ; 
    LOG_IF(fatal, !type_consistent) 
        << " type_consistent FAIL " 
        << " sizeof(T) " << sizeof(T)
        << " a.ebyte " << a->ebyte
        ; 
    assert( type_consistent );  

    //dump(); 
    upload(); 
}

/**
QProp::upload
--------------

1. allocate device array for *nv* T values 
2. populate *prop* on host with device pointers and (height, width) values

   * height is the number of props
   * width is max_num_energy_of_all_prop_plus_one * 2    
     (+1 for integer num_energy last column annotation, as done by NP::combine)

3. copy *pp* array values to device *prop->pp*
4. copy *prop* to device and retain device-side pointer *d_prop*

**/

template<typename T>
void QProp<T>::upload()
{
    prop->pp = QU::device_alloc<T>(nv,"QProp<T>::uploadProps:nv") ; 
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

/**
QProp::lookup
--------------

Note that this is doing separate CUDA launches for each property

**/

template<typename T>
void QProp<T>::lookup( T* lookup, const T* domain,  unsigned num_prop, unsigned domain_width ) const 
{
    unsigned num_lookup = num_prop*domain_width ; 

    LOG(LEVEL) 
        << "["
        << " num_prop " << num_prop
        << " domain_width " << domain_width
        << " num_lookup " << num_lookup
        ; 

    T* d_domain = QU::device_alloc<T>(domain_width, "QProp<T>::lookup:domain_width") ; 
    QU::copy_host_to_device<T>( d_domain, domain, domain_width  ); 

    T* d_lookup = QU::device_alloc<T>(num_lookup,"QProp<T>::lookup:num_lookup") ; 

    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    //configureLaunch( numBlocks, threadsPerBlock, domain_width, 1 ); 

    unsigned threads_per_block = 512u ; 
    QU::ConfigureLaunch1D( numBlocks, threadsPerBlock, domain_width, threads_per_block ); 

    for(unsigned iprop=0 ; iprop < num_prop ; iprop++)
    {
        QProp_lookup(numBlocks, threadsPerBlock, d_prop, d_lookup, d_domain, iprop, domain_width );  
    }

    QU::copy_device_to_host_and_free<T>( lookup, d_lookup, num_lookup ); 
     
    LOG(LEVEL) << "]" ; 
}





/**
lookup_scan
-------------

nx lookups in x0->x1 inclusive for each property yielding nx*qp.ni values.

1. create *x* domain array of shape (nx,) with values in range x0 to x1 
2. create *y* lookup array of shape (qp.ni, nx ) 
3. invoke QProp::lookup collecting *y* lookup values from kernel call 
4. save prop, domain and lookup into fold/reldir

**/

template<typename T>
void QProp<T>::lookup_scan(T x0, T x1, unsigned nx, const char* fold, const char* reldir ) const 
{
    NP* x = NP::Linspace<T>( x0, x1, nx ); 
    NP* y = NP::Make<T>(ni, nx ); 

    lookup(y->values<T>(), x->cvalues<T>(), ni, nx );

    a->save(fold, reldir, "prop.npy"); 
    x->save(fold, reldir, "domain.npy"); 
    y->save(fold, reldir, "lookup.npy"); 

    LOG(info) << "save to " << fold << "/" << reldir  ; 
}








/**
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
**/


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
// quell warning: type attributes ignored after type is already defined [-Wattributes]
template struct QUDARAP_API QProp<float>;
template struct QUDARAP_API QProp<double>;
#pragma GCC diagnostic pop



