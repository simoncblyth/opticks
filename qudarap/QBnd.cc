
#include <cuda_runtime.h>
#include <sstream>

#include "SStr.hh"

#include "scuda.h"
#include "squad.h"

#include "QUDA_CHECK.h"
#include "NP.hh"

#include "QTex.hh"
#include "QBnd.hh"
#include "PLOG.hh"

const plog::Severity QBnd::LEVEL = PLOG::EnvLevel("QBnd", "INFO"); 

const QBnd* QBnd::INSTANCE = nullptr ; 
const QBnd* QBnd::Get(){ return INSTANCE ; }

/**
QBnd::QBnd
------------

Narrows the NP array if wide and creates GPU texture 

**/

QBnd::QBnd(const NP* buf)
    :
    dsrc(buf->ebyte == 8 ? buf : nullptr),
    src(buf->ebyte == 4 ? buf : NP::MakeNarrow(dsrc)),
    tex(MakeBoundaryTex(src))
{
    buf->get_names(bnames) ;  // vector of string from NP metadata

    std::cout << "QBnd::QBnd bnames " << bnames.size() << std::endl ; 
    for(unsigned i=0 ; i < bnames.size() ; i++) std::cout << bnames[i] << std::endl ; 
    assert( bnames.size() > 0 ); 
    INSTANCE = this ; 
} 


std::string QBnd::descBoundary() const
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < bnames.size() ; i++) 
       ss << std::setw(2) << i << " " << bnames[i] << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
} 

unsigned QBnd::getNumBoundary() const
{
    return bnames.size(); 
}

const unsigned QBnd::MISSING = ~0u ; 

unsigned QBnd::getBoundaryIndex(const char* spec) const 
{
    unsigned idx = MISSING ; 
    for(unsigned i=0 ; i < bnames.size() ; i++) 
    {
        if(spec && strcmp(bnames[i].c_str(), spec) == 0) 
        {
            idx = i ; 
            break ; 
        }
    }
    return idx ;  
}

unsigned QBnd::getBoundaryLine(const char* spec, unsigned j) const 
{
    unsigned idx = getBoundaryIndex(spec); 
    bool is_missing = idx == MISSING ; 
    bool is_valid = !is_missing && idx < bnames.size() ;

    if(!is_valid) 
    {
        LOG(error) 
            << " not is_valid " 
            << " spec " << spec
            << " idx " << idx
            << " is_missing " << is_missing 
            << " bnames.size " << bnames.size() 
            ;  
    }

    assert( is_valid ); 
    unsigned line = 4*idx + j ;    
    return line ;  
}

/**
QBnd::getMaterialLine
-----------------------

Searches the bname spec for the *material* name in omat or imat slots, 
returning the first found.  

**/

unsigned QBnd::getMaterialLine( const char* material ) const 
{
    unsigned line = MISSING ; 
    for(unsigned i=0 ; i < bnames.size() ; i++) 
    {
        std::vector<std::string> elem ; 
        SStr::Split(bnames[i].c_str(), '/', elem );  
        const char* omat = elem[0].c_str(); 
        const char* imat = elem[3].c_str(); 

        if(strcmp( material, omat ) == 0 )
        {
            line = i*4 + 0 ; 
            break ; 
        }
        if(strcmp( material, imat ) == 0 )
        {
            line = i*4 + 3 ; 
            break ; 
        }
    }
    return line ; 
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

    LOG(LEVEL) << " buf " << ( buf ? buf->desc() : "-" ) ;  
    assert( nm == 4 ); 

    unsigned nx = nl ;           // wavelength samples
    unsigned ny = ni*nj*nk ;     // total number of properties from all (two) float4 property groups of all (4) species in all (~123) boundaries 

    const float* values = buf->cvalues<float>(); 

    char filterMode = 'L' ; 
    //bool normalizedCoords = false ; 
    bool normalizedCoords = true ; 

    QTex<float4>* btex = new QTex<float4>(nx, ny, values, filterMode, normalizedCoords ) ; 

    quad domainX ; 
    domainX.f.x = buf->get_meta<float>("domain_low",   0.f ); 
    domainX.f.y = buf->get_meta<float>("domain_high",  0.f ); 
    domainX.f.z = buf->get_meta<float>("domain_step",  0.f ); 
    domainX.f.w = buf->get_meta<float>("domain_range", 0.f ); 

    LOG(LEVEL)
        << " domain_low " << std::fixed << std::setw(10) << std::setprecision(3) << domainX.f.x  
        << " domain_high " << std::fixed << std::setw(10) << std::setprecision(3) << domainX.f.y  
        << " domain_step " << std::fixed << std::setw(10) << std::setprecision(3) << domainX.f.z 
        << " domain_range " << std::fixed << std::setw(10) << std::setprecision(3) << domainX.f.w  
        ;

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
    std::string s = ss.str(); 
    return s ; 
}

void QBnd::configureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height )
{
    threadsPerBlock.x = 16 ; 
    threadsPerBlock.y = 16 ; 
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

NP* QBnd::lookup()
{
    unsigned width = tex->width ; 
    unsigned height = tex->height ; 
    unsigned num_lookup = width*height ; 

    NP* out = NP::Make<float>(height, width, 4 ); 

    quad* out_ = (quad*)out->values<float>(); 
    lookup( out_ , num_lookup, width, height ); 

    return out ; 
}

// from QBnd.cu
extern "C" void QBnd_lookup_0(dim3 numBlocks, dim3 threadsPerBlock, cudaTextureObject_t texObj, quad4* meta, quad* lookup, unsigned num_lookup, unsigned width, unsigned height ); 

void QBnd::lookup( quad* lookup, unsigned num_lookup, unsigned width, unsigned height )
{
    LOG(LEVEL) << "[" ; 

    if( tex->d_meta == nullptr )
    {
        tex->uploadMeta();    // TODO: not a good place to do this, needs to be more standard
    }
    assert( tex->d_meta != nullptr && "must QTex::uploadMeta() before lookups" );

    dim3 numBlocks ; 
    dim3 threadsPerBlock ; 
    configureLaunch( numBlocks, threadsPerBlock, width, height ); 

    size_t size = num_lookup*sizeof(quad) ;  

    quad* d_lookup  ;  
    QUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_lookup ), size )); 

    QBnd_lookup_0(numBlocks, threadsPerBlock, tex->texObj, tex->d_meta, d_lookup, num_lookup, width, height );  

    QUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(lookup), d_lookup, size, cudaMemcpyDeviceToHost )); 
    QUDA_CHECK( cudaFree(d_lookup) ); 

    LOG(LEVEL) << "]" ; 
}

void QBnd::dump( quad* lookup, unsigned num_lookup, unsigned edgeitems )
{
    LOG(LEVEL); 
    for(unsigned i=0 ; i < num_lookup ; i++)
    {
        if( i < edgeitems || i > num_lookup - edgeitems)
        {
            quad& props = lookup[i] ;  
            std::cout 
                << std::setw(10) << i 
                << std::setw(10) << std::fixed << std::setprecision(3) << props.f.x 
                << std::setw(10) << std::fixed << std::setprecision(3) << props.f.y
                << std::setw(10) << std::fixed << std::setprecision(3) << props.f.z 
                << std::setw(10) << std::fixed << std::setprecision(3) << props.f.w 
                << std::endl 
                ; 
        }
    }
}

