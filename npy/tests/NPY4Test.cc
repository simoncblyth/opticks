// om-;TEST=NPY4Test om-t

#include "NPY.hpp"
#include "SPack.hh"
#include "OPTICKS_LOG.hh"


void test_make_modulo_selection()
{
    LOG(info); 
    NPY<float>* tr = NPY<float>::make_identity_transforms(20) ; 
    tr->fillIndexFlat(); 
    tr->dump("tr"); 

    NPY<float>* a = NPY<float>::make_modulo_selection(tr, 3, 0 ); 
    NPY<float>* b = NPY<float>::make_modulo_selection(tr, 3, 1 ); 
    NPY<float>* c = NPY<float>::make_modulo_selection(tr, 3, 2 ); 
 
    a->dump("a");  
    b->dump("b");  
    c->dump("c");  

    std::vector<NPYBase*> srcs = {a,b,c} ; 
    NPY<float>* tr2 = NPY<float>::make_interleaved( srcs ); 
    tr2->dump("tr2"); 

    bool dump = true ; 
    unsigned mismatch_items = NPY<float>::compare( tr, tr2, dump ); 
    assert( mismatch_items == 0 ); 

}


void test_write_item_()
{
    LOG(info); 
    NPY<float>* tr = NPY<float>::make_identity_transforms(2) ; 
    tr->fillIndexFlat(); 
    tr->dump("tr"); 

    float a[16] ; 
    float b[16] ; 

    tr->write_item_(a, 0); 
    tr->write_item_(b, 1); 

    std::cout << "[a" << std::endl ;  
    for(unsigned i=0 ; i < 16 ; i++) std::cout << a[i] << " " ; 
    std::cout << "]a" << std::endl ;  
    std::cout << "[b" << std::endl ;  
    for(unsigned i=0 ; i < 16 ; i++) std::cout << b[i] << " " ; 
    std::cout << "]b" << std::endl ;  
}

void test_write_item_big()
{
    LOG(info); 
    NPY<unsigned char>* ab = NPY<unsigned char>::make(2, 512, 1024, 4) ; 
    ab->fillIndexFlat(); 

    NPY<unsigned char>* a = NPY<unsigned char>::make(1, 512, 1024, 4) ; 
    a->zero(); 
    NPY<unsigned char>* b = NPY<unsigned char>::make(1, 512, 1024, 4) ; 
    b->zero(); 
    
    ab->write_item_(a->getBytes(), 0); 
    ab->write_item_(b->getBytes(), 1); 

    //a->dump("a");    makes a mess
    //b->dump("b"); 

    const std::vector<NPYBase*>& srcs = {a,b} ; 

    NPY<unsigned char>* ab2 = NPY<unsigned char>::make_interleaved( srcs ); 
     
    bool dump = true ; 
    assert( NPY<unsigned char>::compare(ab,ab2,dump) == 0 ); 
}



template<typename T>
void test_concat()
{
    NPY<T>* a[3] ; 

    std::vector<const NPYBase*> aa ; 
    for(unsigned i=0 ; i < 3 ; i++)
    {
        //a[i] = NPY<T>::make(1, 512, 1024, 4) ; 
        a[i] = NPY<T>::make(1, 4, 4) ; 
        a[i]->zero(); 
        a[i]->fillIndexFlat();  
        aa.push_back(a[i]); 
    }
   
    NPY<T>* c = NPY<T>::old_concat(aa); 
    const char* c_path = "$TMP/NPY4Test/test_concat/old.npy" ; 
    LOG(info) << " c " << c->getShapeString() << " save: " << c_path ; 
    c->save(c_path); 


    NPY<T>* d = NPY<T>::concat(aa); 
    const char* d_path = "$TMP/NPY4Test/test_concat/new.npy" ; 
    LOG(info) << " d " << d->getShapeString() << " save: " << d_path ; 
    d->save(d_path); 


    assert( NPY<T>::compare(c,d,true) == 0 ); 
}


void test_tvec()
{
    glm::tvec4<unsigned char> tv ; 
    tv.x = 1 ; 
    tv.y = 128 ; 
    tv.z = 255 ; 
    tv.w = 255 ; // 256 gives warning and changes value to 0 

    LOG(info) << " tv " << glm::to_string(tv) ; 


}

void test_getQuad_()
{
    NPY<unsigned char>* a = NPY<unsigned char>::make(64, 4); // 4*64 = 256 
    a->fillIndexFlat(); 
    for(unsigned i=0 ; i < a->getNumItems() ; i++)
    {
        glm::tvec4<unsigned char> q = a->getQuad_(i,0,0); 
    
        unsigned char* v = glm::value_ptr(q) ; 
        unsigned int avalue = SPack::Encode(v, 4); 

        glm::tvec4<unsigned char> q2 ; 
        SPack::Decode( avalue, glm::value_ptr(q2), 4 ); 

        LOG(info) 
            << " i " << i 
            << " q " << glm::to_string(q) 
            << " q2 " << glm::to_string(q2) 
            ; 

        assert( q.x == q2.x ); 
        assert( q.y == q2.y ); 
        assert( q.z == q2.z ); 
        assert( q.w == q2.w ); 

    }
}



void test_setQuad_()
{
    LOG(info); 
    NPY<unsigned char>* a = NPY<unsigned char>::make(64, 4); // 4*64 = 256 
    a->fillIndexFlat(); 
    NPY<unsigned char>* b = NPY<unsigned char>::make(64, 4); 
    b->zero(); 

    for(unsigned i=0 ; i < a->getNumItems() ; i++)
    {
        glm::tvec4<unsigned char> q = a->getQuad_(i); 
        b->setQuad_( q, i); 
    }
    assert( NPY<unsigned char>::compare(a,b,true) == 0 ); 
}


void test_getQuad_crossType()
{
    assert( sizeof(unsigned char)*4 == sizeof(unsigned int)); 
    NPY<unsigned char>* a = NPY<unsigned char>::make(64, 4); // 4*64 = 256 
    a->fillIndexFlat(); 

    NPY<unsigned int>* b = NPY<unsigned int>::make(64) ; 
    b->zero(); 
    assert( b->getNumBytes(0) == a->getNumBytes(0) ); 
    b->read_(a->getBytes()); 

    assert( a->getNumItems() == b->getNumItems() ); 

    for(unsigned i=0 ; i < a->getNumItems() ; i++)
    {
        glm::tvec4<unsigned char> q = a->getQuad_(i); 
        unsigned int avalue = SPack::Encode( glm::value_ptr(q), 4); 
        unsigned int bvalue = b->getValue(i, 0,0 );
        LOG(info) 
            << " i " << i 
            << " q " << glm::to_string(q) 
            << " avalue " << avalue
            << " bvalue " << bvalue
            ; 

        assert( avalue == bvalue );   
    }
}




// reshaping needs to be type aware, so it can do the right thing when cast to a different sized type ?

void test_getQuad_crossType_cast_FAILS_RESHAPE()
{
    assert( sizeof(unsigned char)*4 == sizeof(unsigned int)); 
    NPY<unsigned char>* a = NPY<unsigned char>::make(64, 4); // 4*64 = 256 
    a->fillIndexFlat(); 

    NPY<unsigned int>* b = reinterpret_cast<NPY<unsigned int>*>(a) ; 

    for(unsigned i=0 ; i < a->getNumItems() ; i++)
    {
        a->reshape(64,4); 
        glm::tvec4<unsigned char> q = a->getQuad_(i); 
        unsigned int avalue = SPack::Encode( glm::value_ptr(q), 4); 


        b->reshape(64); 
        unsigned int bvalue = b->getValue(i, 0,0 );
        LOG(info) 
            << " i " << i 
            << " q " << glm::to_string(q) 
            << " avalue " << avalue
            << " bvalue " << bvalue
            ; 

        assert( avalue == bvalue );   
    }
}







int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    //test_make_modulo_selection();  
    //test_write_item_(); 
    //test_write_item_big(); 

    //test_concat<unsigned char>(); 

    //test_tvec(); 
    //test_getQuad_(); 
    //test_setQuad_(); 
    test_getQuad_crossType(); 
    //test_getQuad_crossType_cast_FAILS_RESHAPE(); 

    return 0 ; 
}

// om-;TEST=NPY4Test om-t

