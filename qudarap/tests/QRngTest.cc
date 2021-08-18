
#include "NP.hh"
#include "SPath.hh"
#include "QRng.hh"
#include "OPTICKS_LOG.hh"

const char* FOLD = "/tmp/QRngTest" ; 

/**
test_generate
---------------


**/


template <typename T>
void test_generate( QRng& qr, unsigned num, const char* reldir )
{
    NP* u = NP::Make<T>( num ); 

    unsigned long long skipahead_ = 0ull ; 

    qr.generate<T>(u->values<T>(), num, skipahead_ );

    u->save(FOLD, reldir, "u.npy" ); 

    LOG(info) << "save to " << FOLD << "/" << reldir  ; 
}


/**

num_event

num_item
    eg number of photons within the event

num_value
    number of randoms for the item 

**/

template <typename T>
void test_generate_skipahead( QRng& qr, unsigned num_event, unsigned num_item, unsigned num_value, unsigned skipahead_event_offset, const char* reldir )
{
    NP* uu = NP::Make<T>( num_event, num_item, num_value ); 

    unsigned long long offset = skipahead_event_offset ; 

    for( unsigned i=0 ; i < num_event ; i++)
    {
        unsigned long long event_index = i ; 
        unsigned long long skipahead_ = offset*event_index ; 

        T* target = uu->values<T>() + num_item*num_value*i ; 

        qr.generate<T>( target, num_item, num_value, skipahead_ );
    }

    uu->save(FOLD, reldir, "uu.npy" ); 

    LOG(info) << "save to " << FOLD << "/" << reldir  ; 
}







template <typename T>
void test_generate_2( QRng& qr, unsigned num_event, unsigned num_item, unsigned num_value, const char* reldir )
{
    NP* uu = NP::Make<T>( num_event, num_item, num_value ); 

    for( unsigned i=0 ; i < num_event ; i++)
    {
        unsigned event_idx = i ; 

        T* target = uu->values<T>() + num_item*num_value*i ; 

        qr.generate_2<T>( target, num_item, num_value, event_idx );
    }

    uu->save(FOLD, reldir, "uu.npy" ); 

    LOG(info) << "save to " << FOLD << "/" << reldir  ; 
}






int main(int argc, char** argv)
{   
    OPTICKS_LOG(argc, argv); 

    QRng qr ;   // loads and uploads curandState 

    LOG(info) << qr.desc() ; 

    //test_generate<float>(qr, 1000u, "float" ); 

    unsigned num_event = 10u ; 
    unsigned num_item = 100u ; 
    unsigned num_value = 256u ; 

    // *skipahead_event_offset* would normally be estimate of maximum number of random 
    // values for the items : setting to 1 allows to check are getting the expected offsets into the sequence
    // from "event" to "event"
    // unsigned skipahead_event_offset = 1u ; 
    // test_generate_skipahead<float>(qr, num_event, num_item, num_value, skipahead_event_offset, "float" ); 

    test_generate_2<float>(qr, num_event, num_item, num_value, "float" ); 

    return 0 ; 
}

