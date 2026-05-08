#include <sstream>
#include <algorithm>
#include <cuda_runtime.h>

#include "SLOG.hh"
#include "spath.h"
#include "scuda.h"
#include "squad.h"
#include "sphoton.h"
#include "scerenkov.h"

#include "qrng.h"
#include "qcerenkov.h"

#include "NP.hh"

#include "QUDA_CHECK.h"
#include "QBase.hh"
#include "QCerenkov.hh"
#include "QTex.hh"
#include "QTexMaker.hh"
#include "QTexLookup.hh"
#include "QBnd.hh"
#include "QProp.hh"
#include "QU.hh"
#include "QCK.hh"


const plog::Severity QCerenkov::LEVEL = SLOG::EnvLevel("QCerenkov", "DEBUG");

const QCerenkov* QCerenkov::INSTANCE = nullptr ;
const QCerenkov* QCerenkov::Get(){ return INSTANCE ;  }


//const char* QCerenkov::DEFAULT_FOLD = "$TMP/QCerenkovIntegralTest/test_makeICDF_SplitBin" ;

NP* QCerenkov::Load(const char* fold, const char* name)  // static
{
    const char* path = spath::Resolve(fold, name);
    NP* a = NP::Load(path);
    LOG_IF(fatal, !a) << " failed to load array from path " << path ;
    return a ;
}



#ifdef QCERENKOV_ICDF_OLD
QTex<float4>* QCerenkov::MakeTex(const NP* icdf, char filterMode, bool normalizedCoords ) // static
{
    LOG(info) << " icdf " << icdf << " icdf.sstr " << icdf->sstr() << " filterMode " << filterMode ;
    QTex<float4>* tex = QTexMaker::Make2d_f4(icdf, filterMode, normalizedCoords );
    LOG(info) << " tex " << tex ;
    return tex ;
}
#endif


/**
QCerenkov::MakeInstance
------------------------

HMM : FOR THIS TO WORK IT NEEDS TO DO PART OF WHAT QSim::UploadComponents DOES

**/

qcerenkov* QCerenkov::MakeInstance() // static
{
    const QBase* base = QBase::Get();
    LOG_IF(fatal, base == nullptr) << "QBase must be instanciated before QCerenkov - follow QSim::UploadComponents example" ;
    NP_FATAL_ASSERT(base);

    const QBnd* bnd = QBnd::Get();
    LOG_IF(fatal, bnd == nullptr) << "QBnd must be instanciated before QCerenkov - follow QSim::UploadComponents example" ;
    NP_FATAL_ASSERT(bnd);

    qcerenkov* ck = new qcerenkov ;
    ck->base = base->d_base ;
    ck->bnd = bnd->d_qb ;

#ifdef WITH_PROPCOM
    const QProp<float>* prop = QProp<float>::Get();
    assert(prop);
    ck->prop = prop ? prop->d_prop : nullptr ;
#endif

    return ck ;
}


/**
QCerenkov::QCerenkov
----------------------

HUH: no fold form in use from QSim::UploadComponents

LOOKS LIKE MADE BIG CHANGE TO THE IMPL WITHOUT CLEANING
UP THE OLD ICDF ONE ?
OLD APPROACH used icdf created by QCerenkovIntegralTest


ICDF NOT USED
    qcerenkov.h confirms this

**/

#ifdef QCERENKOV_ICDF_OLD

extern "C" void QCerenkov_old_tex_check(dim3 numBlocks, dim3 threadsPerBlock, unsigned width, unsigned height  );

template <typename T>
extern void QCerenkov_old_lookup(dim3 numBlocks, dim3 threadsPerBlock, cudaTextureObject_t texObj, quad4* meta, T* lookup, unsigned num_lookup, unsigned width, unsigned height  );



QCerenkov::QCerenkov(const char* fold_ )
    :
    fold(fold_),
    icdf_( nullptr ),
    icdf( nullptr ),
    filterMode('P'),    // CAUTION: P: is Point mode with interpolation disabled
    normalizedCoords(true),
    tex(nullptr),
    look(nullptr),
    cerenkov(MakeInstance()),
    d_cerenkov(QU::UploadArray<qcerenkov>(cerenkov, 1, "QCerenkov::QCerenkov/d_cerenkov.1"))
{
    old_init();
}

void QCerenkov::old_init()
{
    INSTANCE = this ;
    assert(fold);

    icdf_ = Load(fold, "icdf.npy");
    if( icdf_ == nullptr )
    {
        LOG(fatal) << "failed to load icdf.npy from fold " << fold ;
        return ;
    }

    LOG(info)
        << " icdf_ " << icdf_
        << " icdf_.sstr " << icdf_->sstr()
        << " icdf_.meta " << icdf_->meta
        ;

    icdf = icdf_->ebyte == 4 ? icdf_ : NP::MakeNarrow(icdf_) ;

    LOG(info)
        << " icdf " << icdf
        << " icdf.sstr " << icdf->sstr()
        << " icdf.meta " << icdf->meta
        ;

    tex = MakeTex(icdf, filterMode, normalizedCoords ) ;

    LOG(info) << " tex " << tex ;

    look = new QTexLookup<float4>( tex ) ;

    LOG(info) << " look " << look ;
}

NP* QCerenkov::old_lookup()
{
    return look ? look->lookup() : nullptr ;
}

void QCerenkov::old_tex_check()
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
    QCerenkov_old_tex_check(numBlocks, threadsPerBlock, width, height );

    cudaDeviceSynchronize();
}






#else
QCerenkov::QCerenkov()
    :
    cerenkov(MakeInstance()),
    d_cerenkov(QU::UploadArray<qcerenkov>(cerenkov, 1,"QCerenkov::QCerenkov/d_cerenkov.0"))
{
    INSTANCE = this ;
}
#endif






std::string QCerenkov::desc() const
{
    std::stringstream ss ;
    ss << "QCerenkov"
#ifdef QCERENKOV_ICDF_OLD
       << " QCERENKOV_ICDF_OLD "
       << " fold " << ( fold ? fold : "-" )
       << " icdf_ " << ( icdf_ ? icdf_->sstr() : "-" )
       << " icdf " << ( icdf ? icdf->sstr() : "-" )
       << " tex " << tex
#else
       << " NOT:QCERENKOV_ICDF_OLD "
#endif
       ;

    std::string s = ss.str();
    return s ;
}






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

