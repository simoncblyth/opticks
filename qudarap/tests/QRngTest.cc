/**
QRngTest.cc
============

TEST=ctor ~/o/qudarap/tests/QRngTest.sh 

OPTICKS_MAX_PHOTON=M4 ~/o/qudarap/tests/QRngTest.sh


**/

#include "NP.hh"
#include "ssys.h"
#include "spath.h"

#include "QRng.hh"
#include "OPTICKS_LOG.hh"

struct QRngTest
{
    typedef unsigned long long ULL ; 
    static constexpr const ULL M = 1000000ull ;
    static constexpr const ULL skipahead_event_offset = 1ull  ;  // small for visibility  
 
    QRng qr ;
    QRngTest(); 

    template <typename T, int MODE> NP* generate_evid( unsigned num_event, unsigned num_item, unsigned num_value ); 

    int ctor();
    int generate_evid();

    static constexpr const char* _SKIPAHEAD = "QRngTest__generate_SKIPAHEAD" ; 
    static constexpr const char* _NV = "QRngTest__generate_NV" ; 

    int generate();
    int main();
};


QRngTest::QRngTest()
    :
    qr()  // may load and upload curandState depending on srng<RNG>::UPLOAD_RNG_STATES
{
    LOG(info) << qr.desc() ; 
} 
   
int QRngTest::ctor()
{
    LOG(info) << qr.desc() ; 
    return 0 ; 
}

/**
QRngTest::generate_evid
-------------------------

num_event

num_item
    eg number of photons within the event

num_value
    number of randoms for the item 

mode:0
    low level manual setting of the skipahead using skipahead_event_offset

mode:1
    slight encapsulation for the event skipaheads

skipahead_event_offset
    would normally be estimate of maximum number of random 
    values for the items : setting to 1 allows to check are getting the expected offsets into the sequence

**/

template <typename T, int MODE>
NP* QRngTest::generate_evid( unsigned num_event, unsigned num_item, unsigned num_value )
{
    NP* uu = NP::Make<T>( num_event, num_item, num_value ); 
    for( unsigned evid=0 ; evid < num_event ; evid++)
    {
        T* target = uu->values<T>() + num_item*num_value*evid ; 
        assert(target); 
        switch(MODE)
        {
           case 0: /*qr.generate<T>(      target, num_item, num_value, skipahead_event_offset*evid )*/ ; break ; 
           case 1: /*qr.generate_evid<T>( target, num_item, num_value, evid  )                      */ ; break ; 
        }
    }
    return uu ; 
}

int QRngTest::generate_evid()
{
    unsigned num_event = 3u ; 
    unsigned num_item = unsigned(qr.rngmax) ; ; 
    unsigned num_value = 16u ; 

    NP* uu = generate_evid<float,1>( num_event, num_item, num_value ) ; 

    uu->save("$FOLD/float", QRng::IMPL, "uu.npy" ); 

    return 0 ; 
}


/**
QRngTest::generate
-------------------

For rngmax M100 and nv:16 this leads to ~6GB output array, 
which was truncated to 2GB until improved NP.hh with NP::INT std::int64_t:: 

    In [2]: 100*1e6*16*4/(1024*1024*1024)
    Out[2]: 5.9604644775390625

::

    OPTICKS_MAX_PHOTON=M200 TEST=generate ~/o/qudarap/tests/QRngTest.sh 

**/

int QRngTest::generate()
{
    unsigned ni = unsigned(qr.rngmax) ; 
    unsigned nv = ssys::getenvunsigned(_NV, 4u) ; 
    unsigned long long skipahead = ssys::getenvull(_SKIPAHEAD, 0ull) ; 

    NP* u = NP::Make<float>( ni, nv ); 

    //qr.generate<float>( u->values<float>(), ni, nv,  skipahead );

    const char* name = sstr::Format("u_%llu.npy", skipahead ); 
    u->save("$FOLD/float", QRng::IMPL, name ); 

    return 0 ; 
}

int QRngTest::main()
{
    const char* TEST = ssys::getenvvar("TEST","generate"); 
    bool ALL = strcmp(TEST, "ALL") == 0 ; 

    std::cout << "[QRngTest::main TEST:[" << TEST << "]\n" ; 

    int rc = 0 ; 
    if(ALL||strcmp(TEST,"ctor")==0)          rc += ctor(); 
    if(ALL||strcmp(TEST,"generate")==0)      rc += generate(); 
    if(ALL||strcmp(TEST,"generate_evid")==0) rc += generate_evid(); 

    std::cout << "]QRngTest::main rc:" << rc << "\n" ; 

    return rc ; 
}

int main(int argc, char** argv)
{   
    OPTICKS_LOG(argc, argv); 
    std::cout 
        << "[main argv[0] " << argv[0] 
        << " QRng::IMPL[" << QRng::IMPL << "]"
        << "\n"
        ; 

    QRngTest t ; 
    int rc = t.main() ; 

    std::cout 
        << "]main argv[0] " << argv[0] 
        << " QRng::IMPL[" << QRng::IMPL << "]"
        << " rc:" << rc 
        << "\n" 
        ; 

    return rc ; 
}

