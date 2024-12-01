/**
QRngTest.cc
============

~/o/qudarap/tests/QRngTest.sh 

**/

#include "NP.hh"
#include "ssys.h"
#include "spath.h"

#include "QRng.hh"
#include "OPTICKS_LOG.hh"

struct QRngTest
{
    QRng qr ;
    QRngTest(); 

    template <typename T> NP* generate_skipahead( int mode, unsigned num_event, unsigned num_item, unsigned num_value, unsigned skipahead_event_offset ); 

    int generate_with_skip();
    int generate();
    int main();
};


QRngTest::QRngTest()
    :
    qr()  // loads and uploads curandState
{
    LOG(info) << "[QRngTest" ; 
    LOG(info) << qr.desc() ; 
    LOG(info) << "]QRngTest" ; 
} 
   

/**
QRngTest::generate_skipahead
-----------------------------

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
NP* QRngTest::generate_skipahead( int mode, unsigned num_event, unsigned num_item, unsigned num_value, unsigned skipahead_event_offset )
{
    NP* uu = NP::Make<T>( num_event, num_item, num_value ); 

    unsigned long long offset = skipahead_event_offset ; 

    for( unsigned i=0 ; i < num_event ; i++)
    {
        unsigned long long event_index = i ; 

        T* target = uu->values<T>() + num_item*num_value*i ; 
        if( mode == 0 )
        {
            unsigned long long skipahead_ = offset*event_index ; 
            qr.generate<T>(   target, num_item, num_value, skipahead_ ) ;
        } 
        else if ( mode == 2 )
        {
            qr.generate_2<T>( target, num_item, num_value, i  ) ; break ; 
        }
    }
    return uu ; 
}

int QRngTest::generate_with_skip()
{
    unsigned num_event = 10u ; 
    unsigned num_item = 100u ; 
    unsigned num_value = 256u ; 
    unsigned skipahead_event_offset = 1u ; 

    int mode = 2 ; 
    NP* uu = generate_skipahead<float>( mode, num_event, num_item, num_value, skipahead_event_offset ) ; 

    uu->save("$FOLD/float", QRng::IMPL, "uu.npy" ); 

    return 0 ; 
}

int QRngTest::generate()
{
    unsigned num = 1000000 ; 
    unsigned long long skipahead_ = 0ull ; 
    NP* u = NP::Make<float>( num ); 
    qr.generate<float>(u->values<float>(), num, 1,  skipahead_ );
    u->save("$FOLD/float", QRng::IMPL, "u.npy" ); 
    return 0 ; 
}

int QRngTest::main()
{
    const char* TEST = ssys::getenvvar("TEST","generate"); 
    bool ALL = strcmp(TEST, "ALL") == 0 ; 

    int rc = 0 ; 
    if(ALL||strcmp(TEST,"generate")==0)           rc += generate(); 
    if(ALL||strcmp(TEST,"generate_with_skip")==0) rc += generate_with_skip(); 

    return rc ; 
}

int main(int argc, char** argv)
{   
    OPTICKS_LOG(argc, argv); 
    QRngTest t ; 
    return t.main() ; 
}

