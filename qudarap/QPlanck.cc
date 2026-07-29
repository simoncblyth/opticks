#include <sstream>
#include <csignal>


#include "SLOG.hh"
#include "ssys.h"
#include "scuda.h"
#include "squad.h"


#include "NP.hh"
#include "QUDA_CHECK.h"
#include "QPlanck.hh"
#include "QTex.hh"
#include "QU.hh"

#include "qplanck.h"


const plog::Severity QPlanck::LEVEL = SLOG::EnvLevel("QPlanck", "DEBUG");

const QPlanck* QPlanck::INSTANCE = nullptr ;
const QPlanck* QPlanck::Get(){ return INSTANCE ;  }

/**
QPlanck::QPlanck
----------------

Canonical invokation from QSim::UploadComponents prior to QSim::QSim

1. Uploads icdf array into GPU texture
2. Creates qplanck instance hooked up with the planck_tex and uploads the instance

**/

QPlanck::QPlanck()
    :
    blackbody(4096,4096,6500.,80.,800.,1'000),
    bb_icdf(blackbody.icdf),
    dsrc(bb_icdf->ebyte == 8 ? bb_icdf : nullptr),
    src( bb_icdf->ebyte == 4 ? bb_icdf : NP::MakeNarrow(dsrc) ),
    tex(MakeTex(src)),
    planck(MakeInstance(tex)),
    d_planck(QU::UploadArray<qplanck>(planck, 1, "QPlanck::QPlanck/d_planck"))
{
    INSTANCE = this ;
}


qplanck* QPlanck::MakeInstance(const QTex<float>* tex) // static
{
    qplanck* planck = new qplanck ;
    planck->tex = tex->texObj ;
    return planck ;
}


std::string QPlanck::desc() const
{
    std::stringstream ss ;
    ss << "QPlanck"
       << " dsrc " << ( dsrc ? dsrc->desc() : "-" )
       << " src " << ( src ? src->desc() : "-" )
       << " tex " << ( tex ? tex->desc() : "-" )
       << " tex " << tex
       ;

    std::string str = ss.str();
    return str ;
}

/**
QPlanck::MakeTex
-----------------------

**/

QTex<float>* QPlanck::MakeTex(const NP* src)  // static
{
    bool expected_shape = src->has_shape(1,4096,1) ;
    LOG_IF(fatal, !expected_shape) << " unexpected shape of src " << ( src ? src->sstr() : "-" ) ;
    assert( expected_shape );

    unsigned ni = src->shape[0];
    unsigned nj = src->shape[1];
    unsigned nk = src->shape[2];

    bool src_expect = src->uifc == 'f' && src->ebyte == 4 && ni == 1 && nj == 4096 && nk == 1  ;
    assert( src_expect );
    if(!src_expect) std::raise(SIGINT);


    unsigned ny = ni ; // height : 1
    unsigned nx = nj ; // width  : 4096


    bool disable_interpolation = ssys::getenvbool("QPLANCK_DISABLE_INTERPOLATION");
    char filterMode = disable_interpolation ? 'P' : 'L' ;

    LOG_IF(fatal, disable_interpolation) << "QPLANCK_DISABLE_INTERPOLATION active using filterMode " << filterMode ;

    bool normalizedCoords = true ;
    QTex<float>* tx = new QTex<float>(nx, ny, src->cvalues<float>(), filterMode, normalizedCoords, src ) ;

    tx->setHDFactor(0);
    tx->uploadMeta();

    LOG(LEVEL)
        << " src " << src->desc()
        << " nx (width) " << nx
        << " ny (height) " << ny
        << " tx.HDFactor " << tx->getHDFactor()
        << " tx.filterMode " << tx->getFilterMode()
        << " tx.normalizedCoords " << tx->getNormalizedCoords()
        ;

    return tx ;
}






/**

extern "C" void QPlanck_check(dim3 numBlocks, dim3 threadsPerBlock, unsigned width, unsigned height  );
extern "C" void QPlanck_lookup(dim3 numBlocks, dim3 threadsPerBlock, cudaTextureObject_t texObj, quad4* meta, float* lookup, unsigned num_lookup, unsigned width, unsigned height  );

void QPlanck::ConfigureLaunch( dim3& numBlocks, dim3& threadsPerBlock, unsigned width, unsigned height )  // static
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



void QPlanck::check() const
{
    unsigned width = tex->width ;
    unsigned height = tex->height ;

    LOG(LEVEL)
        << " width " << width
        << " height " << height
        ;

    dim3 numBlocks ;
    dim3 threadsPerBlock ;
    ConfigureLaunch( numBlocks, threadsPerBlock, width, height );
    QPlanck_check(numBlocks, threadsPerBlock, width, height );

    cudaDeviceSynchronize();
}


NP* QPlanck::lookup() const
{
    unsigned width = tex->width ;
    unsigned height = tex->height ;
    unsigned num_lookup = width*height ;

    LOG(LEVEL)
        << " width " << width
        << " height " << height
        << " lookup " << num_lookup
        ;

    NP* out = NP::Make<float>(height, width, 1 );  // payload dimension of 1 to match source
    float* out_v = out->values<float>();
    lookup( out_v , num_lookup, width, height );

    return out ;
}

void QPlanck::lookup( float* lookup, unsigned num_lookup, unsigned width, unsigned height  ) const
{
    LOG(LEVEL) << "[" ;
    dim3 numBlocks ;
    dim3 threadsPerBlock ;
    ConfigureLaunch( numBlocks, threadsPerBlock, width, height );

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

    QPlanck_lookup(numBlocks, threadsPerBlock, tex->texObj, tex->d_meta, d_lookup, num_lookup, width, height );

    QUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( lookup ), d_lookup, size, cudaMemcpyDeviceToHost ));
    QUDA_CHECK( cudaFree(d_lookup) );

    cudaDeviceSynchronize();

    LOG(LEVEL) << "]" ;
}




void QPlanck::Dump( float* lookup, unsigned num_lookup, unsigned edgeitems  ) // static
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


**/
