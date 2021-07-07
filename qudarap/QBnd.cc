
#include <cuda_runtime.h>
#include "scuda.h"
#include "QUDA_CHECK.h"

#include "NPY.hpp"
#include "GBndLib.hh"
#include "QTex.hh"
#include "QBnd.hh"
#include "PLOG.hh"

const plog::Severity QBnd::LEVEL = PLOG::EnvLevel("QBnd", "INFO"); 

const QBnd* QBnd::INSTANCE = nullptr ; 
const QBnd* QBnd::Get(){ return INSTANCE ; }


QBnd::QBnd(const GBndLib* blib_ )
    :
    blib(blib_),
    dsrc(blib->getBuffer()),
    src(NPY<double>::MakeFloat(dsrc)),
    tex(nullptr)
{
    INSTANCE = this ; 
    init();
} 

void QBnd::init()
{
    makeBoundaryTex(src); 
}

/**
QBnd::makeBoundaryTex
------------------------

Creates GPU texture with material and surface properties as a function of wavelenth.
Example of mapping from 5D array of floats into 2D texture of float4::

    .     ni nj nk nl nm
    blib  36, 4, 2,39, 4

          ni*nk*nk         -> ny  36*4*2 = 288
                   nl      -> nx            39    
                      nm   -> float4 elem    4    

         nx*ny = 11232


TODO: need to get boundary domain range metadata into buffer json sidecar and get it uploaded with the tex

**/

void QBnd::makeBoundaryTex(const NPY<float>* buf )  
{
    unsigned ni = buf->getShape(0);  // (~123) number of boundaries 
    unsigned nj = buf->getShape(1);  // (4)    number of species : omat/osur/isur/imat 
    unsigned nk = buf->getShape(2);  // (2)    number of float4 property groups per species 
    unsigned nl = buf->getShape(3);  // (39 or 761)   number of wavelength samples of the property

    unsigned nm = buf->getShape(4);  // (4)    number of prop within the float4
    LOG(LEVEL) << " buf " << ( buf ? buf->getShapeString() : "-" ) ;  
    assert( nm == 4 ); 

    unsigned nx = nl ;           // wavelength samples
    unsigned ny = ni*nj*nk ;     // total number of properties from all (two) float4 property groups of all (4) species in all (~123) boundaries 

    const float* values = buf->getValuesConst(); 
    
    quad domainX ; 
    // TODO: pass the metadata when do MakeFloat, so do not have to remember to get the meta from the original double buf
    domainX.f.x = dsrc->getMeta<float>("domain_low", "0" ); 
    domainX.f.y = dsrc->getMeta<float>("domain_high", "0" ); 
    domainX.f.z = dsrc->getMeta<float>("domain_step", "0" ); 
    domainX.f.w = dsrc->getMeta<float>("domain_range", "0" ); 

    LOG(LEVEL)
        << " domain_low " << std::fixed << std::setw(10) << std::setprecision(3) << domainX.f.x  
        << " domain_high " << std::fixed << std::setw(10) << std::setprecision(3) << domainX.f.y  
        << " domain_step " << std::fixed << std::setw(10) << std::setprecision(3) << domainX.f.z 
        << " domain_range " << std::fixed << std::setw(10) << std::setprecision(3) << domainX.f.w  
        ;

    assert( domainX.f.y > domainX.f.x ); 
    assert( domainX.f.z > 0.f ); 
    assert( domainX.f.w == domainX.f.y - domainX.f.x ); 

    char filterMode = 'L' ; 
    tex = new QTex<float4>(nx, ny, values, filterMode ) ; 
    tex->setMetaDomainX(&domainX); 
    tex->uploadMeta(); 

}

std::string QBnd::desc() const
{
    std::stringstream ss ; 
    ss << "QBnd"
       << " src " << ( src ? src->getShapeString() : "-" )
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

NPY<float>* QBnd::lookup()
{
    unsigned width = tex->width ; 
    unsigned height = tex->height ; 
    unsigned num_lookup = width*height ; 

    NPY<float>* out = NPY<float>::make(height, width, 4 ); 
    out->zero();  

    quad* out_ = (quad*)out->getValues(); 
    lookup( out_ , num_lookup, width, height ); 

    return out ; 
}


extern "C" void QBnd_lookup_0(dim3 numBlocks, dim3 threadsPerBlock, cudaTextureObject_t texObj, quad4* meta, quad* lookup, unsigned num_lookup, unsigned width, unsigned height ); 

void QBnd::lookup( quad* lookup, unsigned num_lookup, unsigned width, unsigned height )
{
    LOG(LEVEL) << "[" ; 
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

