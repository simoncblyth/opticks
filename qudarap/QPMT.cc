/**
QPMT.cc
==========

QPMT::init
QPMT::init_thickness 
QPMT::init_lcqs
    prep hostside qpmt.h instance and upload to device at d_pmt 

QPMT::lpmtcat_check
    check domain and lookup shape consistency 

QPMT::lpmtcat_
   interface in .cc to kernel launcher QPMT_lpmtcat in .cu   

QPMT::mct_lpmtid_
   interface in .cc to kernel launcher QPMT_mct_lpmtid in .cu   


**/

#include "SLOG.hh"

#if defined(MOCK_CURAND) || defined(MOCK_CUDA)
#else
#include <cuda_runtime.h>
#endif

#include <vector_types.h>
#include "QPMT.hh"


#if defined(MOCK_CURAND) || defined(MOCK_CUDA)
#include "QPMT_MOCK.h"

template<typename T>
const plog::Severity QPMT<T>::LEVEL = plog::info ; 
#else
template<typename T>
const plog::Severity QPMT<T>::LEVEL = SLOG::EnvLevel("QPMT", "DEBUG"); 
#endif

template<typename T>
const QPMT<T>* QPMT<T>::INSTANCE = nullptr ; 

template<typename T>
const QPMT<T>* QPMT<T>::Get(){ return INSTANCE ;  }



template<typename T>
inline std::string QPMT<T>::Desc() // static
{
    std::stringstream ss ; 
    ss << "QPMT<" << ( sizeof(T) == 4 ? "float" : "double" ) << "> " ; 
#ifdef WITH_CUSTOM4
    ss << "WITH_CUSTOM4 " ; 
#else
    ss << "NOT:WITH_CUSTOM4 " ; 
#endif
    ss << " INSTANCE:" << ( INSTANCE ? "YES" : "NO " ) << " " ; 
    if(INSTANCE) ss << INSTANCE->desc() ; 
    std::string str = ss.str(); 
    return str ; 
}

/**
QPMT::init
------------

1. populate hostside qpmt.h instance with device side pointers 
2. upload the hostside qpmt.h instance to GPU
3. retain d_pmt pointer to use in launches

**/

template<typename T>
inline void QPMT<T>::init()
{
    INSTANCE = this ; 

    const int ni = qpmt_NUM_CAT ;   // 3:NNVT/HAMA/NNVT_HiQE
    const int nj = qpmt_NUM_LAYR ;  // 4:Pyrex/ARC/PHC/Vacuum
    const int nk = qpmt_NUM_PROP ;  // 2:RINDEX/KINDEX 
                                    // -----------------
                                    // 3*4*2 = 24

    assert( src_rindex->has_shape(ni, nj, nk, -1, 2 )); 
    assert( src_thickness->has_shape(ni, nj, 1 )); 

    pmt->rindex_prop = rindex_prop->getDevicePtr() ;  
    pmt->qeshape_prop = qeshape_prop->getDevicePtr() ;

    init_thickness(); 
    init_lcqs(); 


#if defined(MOCK_CURAND) || defined(MOCK_CUDA)
    d_pmt = pmt ; 
#else
    d_pmt = QU::UploadArray<qpmt<T>>( (const qpmt<T>*)pmt, 1u, "QPMT::init/d_pmt" ) ;  
    // getting above line to link required template instanciation at tail of qpmt.h 
#endif


}

template<typename T>
inline void QPMT<T>::init_thickness()
{
    const char* label = "QPMT::init_thickness/d_thickness" ; 

#if defined(MOCK_CURAND) || defined(MOCK_CUDA)
    T* d_thickness = const_cast<T*>(thickness->cvalues<T>()) ; 
#else
    T* d_thickness = QU::UploadArray<T>(thickness->cvalues<T>(), thickness->num_values(), label ); ; 
#endif

    pmt->thickness = d_thickness ; 

}

template<typename T>
inline void QPMT<T>::init_lcqs()
{
    LOG(LEVEL) 
       << " src_lcqs " << ( src_lcqs ? src_lcqs->sstr() : "-" )
       << " lcqs " << ( lcqs ? lcqs->sstr() : "-" )
       ;

    const char* label = "QPMT::init_lcqs/d_lcqs" ; 

#if defined(MOCK_CURAND) || defined(MOCK_CUDA)
    T* d_lcqs = lcqs ? const_cast<T*>(lcqs->cvalues<T>()) : nullptr ; 
#else
    T* d_lcqs = lcqs ? QU::UploadArray<T>(lcqs->cvalues<T>(), lcqs->num_values(), label) : nullptr ; 
#endif

    pmt->lcqs = d_lcqs ; 
    pmt->i_lcqs = (int*)d_lcqs ;   // HMM: would cause issues with T=double  
}



#if defined(MOCK_CURAND) || defined(MOCK_CUDA)

template <typename F>
extern void QPMT_lpmtcat_MOCK(
    qpmt<F>* pmt,
    int etype, 
    F* lookup,
    const F* domain,
    unsigned domain_width 
);

template <typename T>
extern void QPMT_mct_lpmtid_MOCK(
    qpmt<T>* pmt,
    int etype, 
    T* lookup,
    const T* domain,
    unsigned domain_width,
    const int* lpmtid, 
    unsigned num_lpmtid 
);


#else
// NB these cannot be extern "C" as need C++ name mangling for template types
template <typename T>
extern void QPMT_lpmtcat(
    dim3 numBlocks,
    dim3 threadsPerBlock,
    qpmt<T>* pmt,
    int etype, 
    T* lookup,
    const T* domain,
    unsigned domain_width
);

template <typename T>
extern void QPMT_mct_lpmtid(
    dim3 numBlocks,
    dim3 threadsPerBlock,
    qpmt<T>* pmt,
    int etype, 
    T* lookup,
    const T* domain,
    unsigned domain_width,
    const int* lpmtid, 
    unsigned num_lpmtid 
);
#endif




template<typename T>
void QPMT<T>::lpmtcat_check( int etype, const NP* domain, const NP* lookup) const 
{
    assert( domain->shape.size() == 1 && domain->shape[0] > 0 ); 
    unsigned num_domain = domain->shape[0] ; 
    unsigned num_domain_1 = 0 ; 

    if( etype == qpmt_RINDEX || etype == qpmt_QESHAPE )
    {
        num_domain_1 = lookup->shape[lookup->shape.size()-1] ; 
    } 
    else if ( etype == qpmt_CATSPEC )
    {
        num_domain_1 = lookup->shape[lookup->shape.size()-3] ;  // (4,4) payload
    }
    assert( num_domain == num_domain_1 ); 
}



/**
QPMT::lpmtcat_
--------------------

1. create hostside lookup array for the output 
2. upload domain array to d_domain
3. allocate lookup array at d_lookup
4. invoke QPMT_lpmtcat launch using d_pmt pointer argument  
5. copy d_lookup to h_lookup 

For some etype the lookup contains an energy_eV scans for all pmt cat (3), 
layers (4) and props (2) (RINDEX, KINDEX). 
So the shape of the lookup output is  (3,4,2, domain_width )   

Those are populated with nested loops (24 props) in the kernel 
with the energy domain passed in as input. Parallelism is over the energy.
 
**/

template<typename T>
NP* QPMT<T>::lpmtcat_(int etype, const NP* domain ) const 
{
    unsigned num_domain = domain->shape[0] ; 
    NP* lookup = MakeArray_lpmtcat(etype, num_domain ); 
    lpmtcat_check(etype, domain, lookup) ; 
    unsigned num_lookup = lookup->num_values() ; 

    const char* label_0 = "QPMT::lpmtcat_/d_domain" ; 

#if defined(MOCK_CURAND) || defined(MOCK_CUDA)
    const T* d_domain = domain->cvalues<T>() ; 
#else
    const T* d_domain = QU::UploadArray<T>( domain->cvalues<T>(), num_domain, label_0 ) ; 
#endif

    LOG(LEVEL) 
        << " etype " << etype 
        << " domain " << domain->sstr()
        << " num_domain " << num_domain 
        << " lookup " << lookup->sstr()
        << " num_lookup " << num_lookup 
        ;

    T* h_lookup = lookup->values<T>() ; 

#if defined(MOCK_CURAND) || defined(MOCK_CUDA)
    T* d_lookup = h_lookup ;  
#else
    T* d_lookup = QU::device_alloc<T>(num_lookup,"QPMT<T>::lpmtcat::d_lookup") ;
#endif   

#if defined(MOCK_CURAND) || defined(MOCK_CUDA)
    QPMT_lpmtcat_MOCK( d_pmt, etype, d_lookup, d_domain, num_domain );  
#else
    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    QU::ConfigureLaunch1D( numBlocks, threadsPerBlock, num_domain, 512u ); 
    
    QPMT_lpmtcat(numBlocks, threadsPerBlock, d_pmt, etype, d_lookup, d_domain, num_domain );

    const char* label_1 = "QPMT::lpmtcat_" ; 
    QU::copy_device_to_host_and_free<T>( h_lookup, d_lookup, num_lookup, label_1 );
    cudaDeviceSynchronize();  

#endif

    return lookup ; 
}


/**
QPMT::mct_lpmtid_
-------------------

mct means minus_cos_theta for an AOI scan 

1. create lookup output array with shape depending on etype
2. allocate d_lookup on device
3. upload domain to d_domain
4. upload lpmtid to d_lpmtid 
5. invoke launch QPMT_mct_lpmtid
6. download d_lookup to h_lookup

**/

template<typename T>
NP* QPMT<T>::mct_lpmtid_(int etype, const NP* domain, const NP* lpmtid ) const 
{
    unsigned num_domain = domain->shape[0] ; 
    unsigned num_lpmtid = lpmtid->shape[0] ; 

    NP* lookup = MakeArray_lpmtid(etype, num_domain, num_lpmtid ); 
    unsigned num_lookup = lookup->num_values() ;


    if( etype == qpmt_ART )
    {
        std::vector<std::pair<std::string, std::string>> kvs = 
        {    
            { "title", "QPMT.title" }, 
            { "brief", "QPMT.brief" }, 
            { "name",  "QPMT.name"  },   
            { "label", "QPMT.label" },
            { "ExecutableName", ExecutableName }
        };   
        lookup->set_meta_kv<std::string>(kvs); 
    }


    LOG(LEVEL) 
        << " etype " << etype 
        << " domain " << domain->sstr()
        << " lpmtid " << lpmtid->sstr()
        << " num_domain " << num_domain 
        << " num_lpmtid " << num_lpmtid 
        << " lookup " << lookup->sstr()
        << " num_lookup " << num_lookup 
        ;

#ifdef WITH_CUSTOM4
    T* h_lookup = lookup->values<T>() ; 

#if defined(MOCK_CURAND) || defined(MOCK_CUDA)
    T* d_lookup = h_lookup ; 
#else
    T* d_lookup = QU::device_alloc<T>(num_lookup,"QPMT<T>::lpmtid::d_lookup") ;
#endif
 
    assert( lpmtid->uifc == 'i' && lpmtid->ebyte == 4 ); 

#if defined(MOCK_CURAND) || defined(MOCK_CUDA)
    const T*   d_domain = domain->cvalues<T>() ; 
    const int* d_lpmtid = lpmtid->cvalues<int>() ; 
#else
    const char* label_0 = "QPMT::mct_lpmtid_/d_domain" ; 
    const char* label_1 = "QPMT::mct_lpmtid_/d_lpmtid" ; 

    const T*   d_domain = QU::UploadArray<T>(   domain->cvalues<T>(),   num_domain, label_0) ; 
    const int* d_lpmtid = QU::UploadArray<int>( lpmtid->cvalues<int>(), num_lpmtid, label_1) ; 
#endif



#if defined(MOCK_CURAND) || defined(MOCK_CUDA)
    QPMT_mct_lpmtid_MOCK( d_pmt, etype, d_lookup, d_domain, num_domain, d_lpmtid, num_lpmtid ); 
#else

    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    QU::ConfigureLaunch1D( numBlocks, threadsPerBlock, num_domain, 512u ); 
   
    QPMT_mct_lpmtid(
        numBlocks, 
        threadsPerBlock, 
        d_pmt, 
        etype, 
        d_lookup, 
        d_domain, 
        num_domain, 
        d_lpmtid, 
        num_lpmtid );

    cudaDeviceSynchronize();  

    const char* label = "QPMT::mct_lpmtid_" ; 
    QU::copy_device_to_host_and_free<T>( h_lookup, d_lookup, num_lookup, label );
    cudaDeviceSynchronize();  
#endif

#else
    LOG(fatal) << " QPMT::mct_lpmtid_ requires compilation WITH_CUSTOM4 " ; 
    assert(0) ; 
#endif

    return lookup ; 
}

// found the below can live in header, when headeronly 
//#pragma GCC diagnostic push
//#pragma GCC diagnostic ignored "-Wattributes"
// quell warning: type attributes ignored after type is already defined [-Wattributes]
template struct QUDARAP_API QPMT<float>;
//template struct QUDARAP_API QPMT<double>;
//#pragma GCC diagnostic pop
 
