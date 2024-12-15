#include <sstream>
#include <cstring>
#include "SLOG.hh"

#include "QRng.hh"
#include "SCurandSpec.h"

#ifdef OLD_MONOLITHIC_CURANDSTATE
#include "SCurandStateMonolithic.hh"
#else
#include "SEventConfig.hh"
#include "SCurandState.h"
#endif

#include "sdirectory.h"
#include "ssys.h"

#include "qrng.h"
#include "srng.h"
#include "QU.hh"

#include "QUDA_CHECK.h"

const plog::Severity QRng::LEVEL = SLOG::EnvLevel("QRng", "DEBUG"); 
const QRng* QRng::INSTANCE = nullptr ; 
const QRng* QRng::Get(){ return INSTANCE ;  }

std::string QRng::Desc() // static
{
    std::stringstream ss ; 
    ss << "QRng::Desc"
       << " IMPL:" << IMPL 
       ;
    std::string str = ss.str() ;
    return str ;  
}


/**
QRng::QRng
------------

QRng instanciation is invoked from QSim::UploadComponents

**/

QRng::QRng(unsigned skipahead_event_offset_)
    :
    RNGNAME(srng<RNG>::NAME),
    UPLOAD_RNG_STATES(srng<RNG>::UPLOAD_RNG_STATES),
    skipahead_event_offset(skipahead_event_offset_),
    seed(0ull),
    offset(0ull),
    SEED_OFFSET(ssys::getenvvar("QRng__SEED_OFFSET")),
    parse_rc(SCurandSpec::ParseSeedOffset(seed, offset, SEED_OFFSET )),
    qr(new qrng<RNG>(seed, offset, skipahead_event_offset)), 
    d_qr(nullptr),
#ifdef OLD_MONOLITHIC_CURANDSTATE
    rngmax(0)
#else
    rngmax(SEventConfig::MaxCurand()),
    cs(nullptr)
#endif
{
    init(); 
}



template<> void QRng::initStates<Philox>()
{ 
    LOG(info)
        << "initStates<Philox> DO NOTHING : No LoadAndUpload needed " 
        << " rngmax " << rngmax 
        << " SEventConfig::MaxCurand " << SEventConfig::MaxCurand()
        ; 
}

template<> void QRng::initStates<XORWOW>()
{ 
    bool is_XORWOW = strcmp( srng<XORWOW>::NAME, "XORWOW") == 0 ; 
    assert( is_XORWOW ); 

    LOG(info) << "initStates<XORWOW> LoadAndUpload and set_uploaded_states " ; 
#ifdef OLD_MONOLITHIC_CURANDSTATE
    XORWOW* d_uploaded_states = LoadAndUpload(rngmax, SCurandStateMonolithic::Path()) ;  
#else
    XORWOW* d_uploaded_states = LoadAndUpload(rngmax, cs); 
#endif
    qr->set_uploaded_states( d_uploaded_states ); 
}



void QRng::init()
{
    INSTANCE = this ; 
    assert(parse_rc == 0 ); 

    initStates<RNG>(); 
    initMeta(); 

    bool VERBOSE = ssys::getenvbool(init_VERBOSE); 
    LOG_IF(info, VERBOSE)
         << "[" << init_VERBOSE << "] " << ( VERBOSE ? "YES" : "NO " )
         << "\n"
         << desc()
         ;  
}





/**
QRng::initMeta
------------------

1. record device pointer qr->rng_startes

2. upload qrng.h *qr* instance within single element array, setting d_qr

**/

void QRng::initMeta()
{
    const char* label_1 = "QRng::initMeta/d_qr" ; 
    d_qr = QU::UploadArray<qrng<RNG>>(qr, 1, label_1 ); 

    bool uploaded = d_qr != nullptr ; 
    LOG_IF(fatal, !uploaded) << " FAILED to upload RNG and/or metadata " ;  
    assert(uploaded); 
}



QRng::~QRng()
{
}



#ifdef OLD_MONOLITHIC_CURANDSTATE

const char* QRng::Load_FAIL_NOTES = R"(
QRng::Load_FAIL_NOTES
=================================

QRng::Load failed to load the RNG files. 
These files should have been created during the *opticks-full* installation 
by the bash function *opticks-prepare-installation* 
which runs *qudarap-prepare-installation*. 

Investigate by looking at the contents of the RNG directory, 
as shown below::

    epsilon:~ blyth$ ls -l  ~/.opticks/rngcache/RNG/
    total 892336
    -rw-r--r--  1 blyth  staff   44000000 Oct  6 19:43 QCurandState_1000000_0_0.bin
    -rw-r--r--  1 blyth  staff  132000000 Oct  6 19:53 QCurandState_3000000_0_0.bin
    epsilon:~ blyth$ 


)" ;

#else
const char* QRng::Load_FAIL_NOTES = R"(
QRng::Load_FAIL_NOTES
===============================

TODO : for new chunked impl

)" ;

#endif




#ifdef OLD_MONOLITHIC_CURANDSTATE

/**
QRng::LoadAndUpload
--------------------

In the old monolithic impl rngmax is an output argument obtained from file_size/item_size 
and at the same time kinda an input to specify which file to load. 

In the new chunked impl with partial chunk loading the rngmax is an input value
that can be set to anything. 

**/

XORWOW* QRng::LoadAndUpload(ULL& rngmax, const char* path)  // static 
{
    XORWOW* h_states = Load(rngmax, path); 
    XORWOW* d_states = UploadAndFree(h_states, rngmax ); 
    return d_states ; 
}

XORWOW* QRng::Load(ULL& rngmax, const char* path)  // static 
{
    bool null_path = path == nullptr ; 
    LOG_IF(fatal, null_path ) << " QRng::Load null path " ; 
    assert( !null_path );  

    FILE *fp = fopen(path,"rb");
    bool failed = fp == nullptr ; 
    LOG_IF(fatal, failed ) << " unabled to open file [" << path << "]" ; 
    LOG_IF(error, failed ) << Load_FAIL_NOTES  ; 
    assert(!failed); 


    fseek(fp, 0L, SEEK_END);
    long file_size = ftell(fp);
    rewind(fp);

    long type_size = sizeof(RNG) ;  
    long item_size = 44 ; 

    rngmax = file_size/item_size ; 


    LOG(LEVEL) 
        << " path " << path 
        << " file_size " << file_size 
        << " item_size " << item_size 
        << " type_size " << type_size 
        << " rngmax " << rngmax
        ; 

    assert( file_size % item_size == 0 );  

    XORWOW* rng_states = (XORWOW*)malloc(sizeof(XORWOW)*rngmax);

    for(ULL i = 0 ; i < rngmax ; ++i )
    {   
        XORWOW& rng = rng_states[i] ;
        fread(&rng.d,                     sizeof(unsigned int),1,fp);   //  1
        fread(&rng.v,                     sizeof(unsigned int),5,fp);   //  5 
        fread(&rng.boxmuller_flag,        sizeof(int)         ,1,fp);   //  1 
        fread(&rng.boxmuller_flag_double, sizeof(int)         ,1,fp);   //  1
        fread(&rng.boxmuller_extra,       sizeof(float)       ,1,fp);   //  1
        fread(&rng.boxmuller_extra_double,sizeof(double)      ,1,fp);   //  2    11*4 = 44 
    }   
    fclose(fp);

    return rng_states ; 
}

XORWOW* QRng::UploadAndFree(XORWOW* h_states, ULL num_states )  // static 
{
    const char* label_0 = "QRng::UploadAndFree/rng_states" ; 
    XORWOW* d_states = QU::UploadArray<XORWOW>(h_states, num_states, label_0 ) ;   
    free(h_states); 
    return d_states ;  
}

#else

/**
QRng::LoadAndUpload
----------------------

TODO : replace this, using SCurandState::loadAndUpload


rngmax
    input argument that determines how many chunks of RNG to load and upload

(SCurandState)cs
    vector of SCurandChunk metadata on the chunk files 


For example with chunks of 10M each and rngmax of 25M::

     10M     10M      10M
   +------+--------+-------+
   

Read full chunks until doing so would go over rngmax, then 



RNG load bytes digest 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    QRng::LoadAndUpload complete YES rngmax/M 3 rngmax 3000000 digest c5a80f522e9393efe0302b916affda06


If rngmax lands on a border between files/chunks then the RNG load digest
should match the output from md5sum on the corresponding state files. 
For chunks it is necessary to concat the files first::

    P[blyth@localhost RNG]$ md5sum QCurandStateMonolithic_3000000_0_0.bin
    c5a80f522e9393efe0302b916affda06  QCurandStateMonolithic_3000000_0_0.bin

    P[blyth@localhost RNG]$ cat SCurandChunk_0000_0000M_0001M_0_0.bin SCurandChunk_0001_0001M_0001M_0_0.bin SCurandChunk_0002_0002M_0001M_0_0.bin > /tmp/3M.bin
    P[blyth@localhost RNG]$ md5sum /tmp/3M.bin
    c5a80f522e9393efe0302b916affda06  /tmp/3M.bin

    ## cat SCurandChunk_000[0-2]_00*M_0001M_0_0.bin > /tmp/3M.bin  ## wildcard way 


Note that sizeof(RNG) is slightly larger than the itemsize in the file, 
indicating that RNG in memory has some padding. Due to this digests of 
the RNG in memory do not match those of the files or the loaded bytes.    


rethink auto rngmax:0
~~~~~~~~~~~~~~~~~~~~~~~

While implementing multiple launch running realize that 
reproducibility requires RNG "ix" slot offsetting 
for launches beyond the first. This should allow results from 
multiple launches to exactly match unsplit launches.   

Initially thought that would entail re-uploading the 
states. But it would be simpler to upload all the available 
states at initialization and just offset for each launch.  
Note this is "vertical" picking the slot, not "horizontal" 
offsetting for the skipahead done from event to event.  

While this means need VRAM for the states it looks likely 
that will soon jump to Philox counter based RNG, which will 
remove the need for loading states.  Offsetting of counters
appropriately will still be needed. 

rngmax:0
   load all available states, 
rngmax>0 
   load specified number of states



**/

XORWOW* QRng::LoadAndUpload(ULL _rngmax, const SCurandState& cs)  // static 
{
    LOG(LEVEL) << cs.desc() ; 

    ULL tot_available_states = cs.all.num ; 
    ULL rngmax = _rngmax > 0 ? _rngmax : tot_available_states ; 

    LOG_IF(error, _rngmax == 0 ) 
        << "\n" 
        << " WARNING : _rngmax is ZERO : will load+upload all SCurandChunk files "
        << " consuming significant VRAM and enabling very large launches "
        << " set [" << SEventConfig::kMaxCurand << "] non-zero eg M3 to control "
        << " tot_available_states/M " << tot_available_states/M 
        << " rngmax/M " << rngmax/M
        ;

    XORWOW* d0 = QU::device_alloc<XORWOW>( rngmax, "QRng::LoadAndUpload/rngmax" ); 
    XORWOW* d = d0 ; 

    ULL available_chunk = cs.chunk.size(); 
    ULL count = 0 ; 

    LOG(LEVEL)
        << " rngmax " << rngmax
        << " rngmax/M " << rngmax/M
        << " available_chunk " << available_chunk 
        << " cs.all.num/M " << cs.all.num/M 
        << " tot_available_states/M " << tot_available_states/M 
        << " rngmax/M " << rngmax/M
        << " d0 " << d0 
        ;


    sdigest dig ; 

    for(ULL i=0 ; i < available_chunk ; i++)
    {
        ULL remaining = rngmax - count ;  

        const SCurandChunk& chunk = cs.chunk[i]; 
 
        bool partial_read = remaining < chunk.ref.num ;  

        ULL num = partial_read ? remaining : chunk.ref.num ;

        LOG(LEVEL)
            << " i " << std::setw(3) << i 
            << " chunk.ref.num/M " << std::setw(4) << chunk.ref.num/M
            << " count/M " << std::setw(4) << count/M
            << " remaining/M " << std::setw(4) << remaining/M
            << " partial_read " << ( partial_read ? "YES" : "NO " )
            << " num/M " << std::setw(4) << num/M
            << " d " << d 
            ;

        scurandref<XORWOW> cr = chunk.load(num, cs.dir, &dig ) ;
  
        assert( cr.states != nullptr); 

        bool num_match = cr.num == num ; 

        LOG_IF(fatal, !num_match)
            << "QRng::LoadAndUpload"
            << " num_match " << ( num_match ? "YES" : "NO " )
            << " cr.num/M " << cr.num/M
            << " num/M " << num/M
            ;

        assert(num_match); 

        QU::copy_host_to_device<XORWOW>( d , cr.states , num ); 

        free(cr.states); 

        d += num ;  
        count += num ;  

        if(count > rngmax) assert(0); 
        if(count == rngmax) break ;
    }

    bool complete = count == rngmax ; 
    assert( complete );
    std::string digest = dig.finalize(); 

    std::cout 
        << "QRng::LoadAndUpload"
        << " complete " << ( complete ? "YES" : "NO ")
        << " rngmax/M " << rngmax/M 
        << " rngmax " << rngmax
        << " digest " << digest 
        << "\n"
        ;

    return complete ? d0 : nullptr ; 
}

#endif


/**
QRng::Save
------------

Used from the old QCurandState::save

TODO: eliminate, functionality duplicates in SCurandChunk::Save

**/
void QRng::Save( XORWOW* states, unsigned num_states, const char* path ) // static
{
    sdirectory::MakeDirsForFile(path);
    FILE *fp = fopen(path,"wb");
    LOG_IF(fatal, fp == nullptr) << " error opening file " << path ; 
    assert(fp); 

    for(unsigned i = 0 ; i < num_states ; ++i )
    {   
        XORWOW& rng = states[i] ;
        fwrite(&rng.d,                     sizeof(unsigned int),1,fp);
        fwrite(&rng.v,                     sizeof(unsigned int),5,fp);
        fwrite(&rng.boxmuller_flag,        sizeof(int)         ,1,fp);
        fwrite(&rng.boxmuller_flag_double, sizeof(int)         ,1,fp);
        fwrite(&rng.boxmuller_extra,       sizeof(float)       ,1,fp);
        fwrite(&rng.boxmuller_extra_double,sizeof(double)      ,1,fp);
    }   
    fclose(fp);
    return ; 
}





std::string QRng::desc() const
{
    std::stringstream ss ; 
    ss << "QRng::desc\n"
       << std::setw(30) << " IMPL " << IMPL << "\n" 
       << std::setw(30) << " RNGNAME " << ( RNGNAME ? RNGNAME : "-" ) << "\n" 
       << std::setw(30) << " UPLOAD_RNG_STATES " << ( UPLOAD_RNG_STATES ? "YES" : "NO " ) << "\n"
       << std::setw(30) << " seed " << seed << "\n"
       << std::setw(30) << " offset " << offset << "\n"
       << std::setw(30) << " rngmax " << rngmax << "\n"
       << std::setw(30) << " rngmax/M " << rngmax/M << "\n"
       << std::setw(30) << " qr " << qr << "\n"
       << std::setw(30) << " qr.skipahead_event_offset " << qr->skipahead_event_offset << "\n"
       << std::setw(30) << " d_qr " << d_qr << "\n"
       ;

    std::string str = ss.str(); 
    return str ; 
}





template <typename T>
extern void QRng_generate(
    dim3, 
    dim3, 
    qrng<RNG>*, 
    unsigned, 
    T*, 
    unsigned, 
    unsigned );


/**
QRng::generate
-----------------

Launch ni threads to generate ni*nv values, via [0:nv] loop in the kernel 
with some light touch encapsulation using event_idx to automate skipahead. 

**/


template<typename T>
void QRng::generate( T* uu, unsigned ni, unsigned nv, unsigned evid )
{
    const char* label = "QRng::generate:ni*nv" ; 

    T* d_uu = QU::device_alloc<T>(ni*nv, label );

    QU::ConfigureLaunch(numBlocks, threadsPerBlock, ni, 1 );  

    QRng_generate<T>(numBlocks, threadsPerBlock, d_qr, evid, d_uu, ni, nv ); 

    QU::copy_device_to_host_and_free<T>( uu, d_uu, ni*nv, label );
}


template void QRng::generate<float>( float*,   unsigned, unsigned, unsigned ); 
template void QRng::generate<double>( double*, unsigned, unsigned, unsigned ); 


