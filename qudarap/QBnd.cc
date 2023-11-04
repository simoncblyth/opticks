
#include <cuda_runtime.h>
#include <sstream>
#include <map>

#include "SBnd.h"
#include "NP.hh"
#include "NPFold.h"

#include "scuda.h"
#include "squad.h"
#include "sstate.h"

#if defined(MOCK_TEXTURE) || defined(MOCK_CUDA)
#else
#include "QUDA_CHECK.h"
#include "QU.hh"
#include "SLOG.hh"
#endif

#include "QTex.hh"
#include "QOptical.hh"
#include "QBnd.hh"

#include "qbnd.h"



#if defined(MOCK_TEXTURE) || defined(MOCK_CUDA)
#else
const plog::Severity QBnd::LEVEL = SLOG::EnvLevel("QBnd", "DEBUG"); 
#endif

const QBnd* QBnd::INSTANCE = nullptr ; 
const QBnd* QBnd::Get(){ return INSTANCE ; }

/**
QBnd::MakeInstance
---------------------

static method used from QBnd::QBnd using the bnd array spec names

**/

qbnd* QBnd::MakeInstance(const QTex<float4>* tex, const std::vector<std::string>& names )
{
    qbnd* qb = new qbnd ; 

    qb->boundary_tex = tex->texObj ; 
    qb->boundary_meta = tex->d_meta ; 
    qb->boundary_tex_MaterialLine_Water = SBnd::GetMaterialLine("Water", names) ; 
    qb->boundary_tex_MaterialLine_LS    = SBnd::GetMaterialLine("LS", names) ; 

    const QOptical* optical = QOptical::Get() ; 
    //assert( optical ); 

#if defined(MOCK_TEXTURE) || defined(MOCK_CUDA)
#else
    LOG(LEVEL) << " optical " << ( optical ? optical->desc() : "MISSING" ) ; 
#endif

    qb->optical = optical ? optical->d_optical : nullptr ; 

    assert( qb->optical != nullptr ); 
    assert( qb->boundary_meta != nullptr ); 
    return qb ; 
}


/**
QBnd::QBnd
------------

Narrows the NP array if wide and creates GPU texture 

**/

QBnd::QBnd(const NP* buf)
    :
    dsrc(buf->ebyte == 8 ? buf : nullptr),
    src(NP::MakeNarrowIfWide(buf)),
    sbn(new SBnd(src)),
    tex(MakeBoundaryTex(src)),
    qb(MakeInstance(tex, buf->names)),
    d_qb(nullptr)
{
    init(); 
} 

void QBnd::init()
{
    INSTANCE = this ; 
#if defined(MOCK_TEXTURE) || defined(MOCK_CUDA)
    d_qb = qb ;  
#else
    d_qb = QU::UploadArray<qbnd>(qb,1,"QBnd::QBnd/d_qb") ; 
#endif
}


/**
QBnd::MakeBoundaryTex
------------------------

Creates GPU texture with material and surface properties as a function of wavelenth.
Example of mapping from 5D array of floats into 2D texture of float4::

    .     ni nj nk  nl nm
    blib  36, 4, 2,761, 4

          ni : boundaries
          nj : 0:omat/1:osur/2:isur/3:imat  
          nk : 0 or 1 property group
          nl :  



          ni*nk*nk         -> ny  36*4*2 = 288
                   nl      -> nx           761 (fine domain, 39 when using coarse domain)
                      nm   -> float4 elem    4    

         nx*ny = 11232


TODO: need to get boundary domain range metadata into buffer json sidecar and get it uploaded with the tex

**/

QTex<float4>* QBnd::MakeBoundaryTex(const NP* buf )   // static 
{
    assert( buf->uifc == 'f' && buf->ebyte == 4 );  

    unsigned ni = buf->shape[0];  // (~123) number of boundaries 
    unsigned nj = buf->shape[1];  // (4)    number of species : omat/osur/isur/imat 
    unsigned nk = buf->shape[2];  // (2)    number of float4 property groups per species 
    unsigned nl = buf->shape[3];  // (39 or 761)   number of wavelength samples of the property
    unsigned nm = buf->shape[4];  // (4)    number of prop within the float4

#if defined(MOCK_TEXTURE) || defined(MOCK_CUDA)
#else
    LOG(LEVEL) << " buf " << ( buf ? buf->desc() : "-" ) ;  
#endif
    assert( nm == 4 ); 

    unsigned nx = nl ;           // wavelength samples
    unsigned ny = ni*nj*nk ;     
    // ny : total number of properties from all (two) float4 property 
    // groups of all (4) species in all (~123) boundaries 

    const float* values = buf->cvalues<float>(); 

    char filterMode = 'L' ; 
    //bool normalizedCoords = false ; 
    bool normalizedCoords = true ; 

    QTex<float4>* btex = new QTex<float4>(nx, ny, values, filterMode, normalizedCoords, buf ) ; 

    bool buf_has_meta = buf->has_meta() ;

#if defined(MOCK_TEXTURE) || defined(MOCK_CUDA)
#else
    LOG_IF(fatal, !buf_has_meta) << " buf_has_meta FAIL : domain metadata is required to create texture  buf.desc " << buf->desc() ;  
#endif
    assert( buf_has_meta ); 

    quad domainX ; 
    domainX.f.x = buf->get_meta<float>("domain_low",   0.f ); 
    domainX.f.y = buf->get_meta<float>("domain_high",  0.f ); 
    domainX.f.z = buf->get_meta<float>("domain_step",  0.f ); 
    domainX.f.w = buf->get_meta<float>("domain_range", 0.f ); 

#if defined(MOCK_TEXTURE) || defined(MOCK_CUDA)
#else
    LOG(LEVEL)
        << " domain_low " << std::fixed << std::setw(10) << std::setprecision(3) << domainX.f.x  
        << " domain_high " << std::fixed << std::setw(10) << std::setprecision(3) << domainX.f.y  
        << " domain_step " << std::fixed << std::setw(10) << std::setprecision(3) << domainX.f.z 
        << " domain_range " << std::fixed << std::setw(10) << std::setprecision(3) << domainX.f.w  
        ;
#endif

    assert( domainX.f.y > domainX.f.x ); 
    assert( domainX.f.z > 0.f ); 
    assert( domainX.f.w == domainX.f.y - domainX.f.x ); 

    btex->setMetaDomainX(&domainX); 
    btex->uploadMeta(); 

    return btex ; 
}

std::string QBnd::desc() const
{
    std::stringstream ss ; 
    ss << "QBnd"
       << " src " << ( src ? src->desc() : "-" )
       << " tex " << ( tex ? tex->desc() : "-" )
       << " tex " << tex 
       ; 
    std::string str = ss.str(); 
    return str ; 
}

std::string QBnd::DescLaunch( const dim3& numBlocks, const dim3& threadsPerBlock, int width, int height ) // static
{
    std::stringstream ss ; 
    ss
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

    std::string str = ss.str(); 
    return str ; 
}


void QBnd::ConfigureLaunch( dim3& numBlocks, dim3& threadsPerBlock, int width, int height ) // static 
{
    threadsPerBlock.x = 16 ; 
    threadsPerBlock.y = 16 ; 
    threadsPerBlock.z = 1 ; 
 
    numBlocks.x = (width + threadsPerBlock.x - 1) / threadsPerBlock.x ; 
    numBlocks.y = (height + threadsPerBlock.y - 1) / threadsPerBlock.y ;
    numBlocks.z = 1 ; 
}

NP* QBnd::lookup() const 
{
    unsigned width = tex->width ; 
    unsigned height = tex->height ; 
    unsigned num_lookup = width*height ; 

    NP* out = NP::Make<float>(height, width, 4 ); 

    quad* out_ = (quad*)out->values<float>(); 
    lookup( out_ , num_lookup, width, height ); 

    out->reshape(src->shape); 

    return out ; 
}

NPFold* QBnd::serialize() const 
{
    NPFold* f = new NPFold ; 
    f->add("src", src ); 
    f->add("dst", lookup() ); 
    return f ; 
}

void QBnd::save(const char* dir) const 
{
    NPFold* f = serialize(); 
    f->save(dir); 
}

#if defined(MOCK_TEXTURE) || defined(MOCK_CUDA)

extern "C" void QBnd_lookup_0_MOCK(
    cudaTextureObject_t texObj, 
    quad4* meta, 
    quad* lookup, 
    int num_lookup, 
    int width, 
    int height 
    ); 

#include "QBnd_MOCK.h"

#else

// from QBnd.cu
extern "C" void QBnd_lookup_0(
    dim3 numBlocks, 
    dim3 threadsPerBlock, 
    cudaTextureObject_t texObj, 
    quad4* meta, 
    quad* lookup, 
    int num_lookup, 
    int width, 
    int height 
    ); 

#endif


void QBnd::lookup( quad* lookup, int num_lookup, int width, int height ) const 
{
    if( tex->d_meta == nullptr )
    {
        tex->uploadMeta();    // TODO: not a good place to do this, needs to be more standard
    }
    assert( tex->d_meta != nullptr && "must QTex::uploadMeta() before lookups" );


#if defined(MOCK_TEXTURE) || defined(MOCK_CUDA)

    std::cout << "QBnd::lookup MISSING MOCK IMPL " << std::endl ; 
    quad* d_lookup  = lookup ; 

    QBnd_lookup_0_MOCK(tex->texObj, tex->d_meta, d_lookup, num_lookup, width, height );  

#else

    // TODO: update the below to use more contemporary approach, starting with using QU 

    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    ConfigureLaunch( numBlocks, threadsPerBlock, width, height ); 
    std::cout << DescLaunch( numBlocks, threadsPerBlock, width, height ) << std::endl ; 
    size_t size = num_lookup*sizeof(quad) ;  


    quad* d_lookup  ;  
    QUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_lookup ), size )); 

    QBnd_lookup_0(numBlocks, threadsPerBlock, tex->texObj, tex->d_meta, d_lookup, num_lookup, width, height );  

    QUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(lookup), d_lookup, size, cudaMemcpyDeviceToHost )); 
    QUDA_CHECK( cudaFree(d_lookup) ); 

#endif

}

std::string QBnd::Dump( quad* lookup, int num_lookup, int edgeitems ) // static 
{
    std::stringstream ss ; 

    for(int i=0 ; i < num_lookup ; i++)
    {
        if( i < edgeitems || i > num_lookup - edgeitems)
        {
            quad& props = lookup[i] ;  
            ss
                << std::setw(10) << i 
                << std::setw(10) << std::fixed << std::setprecision(3) << props.f.x 
                << std::setw(10) << std::fixed << std::setprecision(3) << props.f.y
                << std::setw(10) << std::fixed << std::setprecision(3) << props.f.z 
                << std::setw(10) << std::fixed << std::setprecision(3) << props.f.w 
                << std::endl 
                ; 
        }
    }
    std::string str = ss.str(); 
    return str ; 
}

