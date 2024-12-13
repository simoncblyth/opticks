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
    using ULL = unsigned long long ; 
    static constexpr const ULL M = 1000000ull ;

    static constexpr const char* _NUM_EVENT = "QRngTest__NUM_EVENT" ; 
    static constexpr const char* _NUM_VALUE = "QRngTest__NUM_VALUE" ; 
    static constexpr const char* _SKIPAHEAD = "QRngTest__SKIPAHEAD" ; 

    ULL num_event ; 
    ULL num_value ; 
    ULL skipahead_event_offset ;  

    QRng qr ;

    QRngTest(); 


    template <typename T> NP* generate( unsigned num_event, unsigned num_item, unsigned num_value ); 

    int ctor();
    int generate();

    static constexpr const char* _NV = "QRngTest__generate_NV" ; 

    int main();
    static int Main(int argc, char** argv);
};


QRngTest::QRngTest()
    :
    num_event(ssys::getenvull(_NUM_EVENT, 3ull)), 
    num_value(ssys::getenvull(_NUM_VALUE, 16ull)), 
    skipahead_event_offset(ssys::getenvull(_SKIPAHEAD, 1ull)),
    qr(skipahead_event_offset)  // may load and upload curandState depending on srng<RNG>::UPLOAD_RNG_STATES
{
} 
   
int QRngTest::ctor()
{
    LOG(info) << qr.desc() ; 
    return 0 ; 
}

/**
QRngTest::generate
--------------------

num_event

num_item
    eg number of photons within the event

num_value
    number of randoms for the item 

skipahead_event_offset
    would normally be estimate of maximum number of random 
    values for the items : setting to 1 allows to check are getting the expected offsets into the sequence

**/

template <typename T>
NP* QRngTest::generate( unsigned num_event, unsigned num_item, unsigned num_value )
{
    NP* uu = NP::Make<T>( num_event, num_item, num_value );
 
    for( unsigned ev=0 ; ev < num_event ; ev++)
    {
        T* target = uu->values<T>() + num_item*num_value*ev ; 
        assert(target); 
        qr.generate<T>( target, num_item, num_value, ev  );  
    }
    return uu ; 
}


/**
QRngTest::generate
-------------------

**/

int QRngTest::generate()
{
    unsigned num_item = unsigned(qr.rngmax) ; ; 

    NP* uu = generate<float>( num_event, num_item, num_value ) ; 

    uu->save("$FOLD/float", QRng::IMPL, "uu.npy" ); 

    return 0 ; 
}


int QRngTest::main()
{
    const char* TEST = ssys::getenvvar("TEST","generate"); 
    bool ALL = strcmp(TEST, "ALL") == 0 ; 

    LOG(info) << "[TEST:" << TEST ; 

    int rc = 0 ; 
    if(ALL||strcmp(TEST,"ctor")==0)          rc += ctor(); 
    if(ALL||strcmp(TEST,"generate")==0)      rc += generate(); 

    LOG(info) << "]TEST:" << TEST << " rc " << rc  ; 
    return rc ; 
}


int QRngTest::Main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    QRngTest t ; 
    return t.main() ; 
}

int main(int argc, char** argv){ return QRngTest::Main(argc, argv) ; }   

