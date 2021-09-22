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
    float epsilon = 1e-6 ; 
    unsigned mismatch_items = NPY<float>::compare( tr, tr2, epsilon, dump ); 
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
    unsigned char eps = 0; 
    assert( NPY<unsigned char>::compare(ab,ab2,eps, dump) == 0 ); 
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

void test_getQuad_2()
{
    LOG(info); 

    unsigned ni = 10 ; 
    unsigned offset = (0x1 << 24) - 5 ; 

    NPY<unsigned>* u = NPY<unsigned>::make(ni,4); 
    u->fillIndexFlat(offset); 

    NPY<float>* f = NPY<float>::make(ni,4); 
    f->fillIndexFlat(offset); 

    LOG(info) << " glm::uvec4  q = u->getQuad_(i,0,0) : all ok  " ; 
    for(unsigned i=0 ; i < ni ; i++)
    {
        glm::uvec4  q = u->getQuad_(i,0,0);     // same as above 
        std::cout << i << " " << glm::to_string(q) << std::endl ;        
    }

    LOG(info) << " glm::uvec4  q = u->getQuadF(i,0,0)  : LSB truncation bug " ; 
    for(unsigned i=0 ; i < ni ; i++)
    {
        glm::uvec4  q = u->getQuadF(i,0,0);   
        std::cout << i << " " << glm::to_string(q) << std::endl ;        
    }

    LOG(info) << " glm::vec4 q = f->getQuadF(i,0,0) : notice the floats canna take it up here  " ; 
    for(unsigned i=0 ; i < ni ; i++)
    {
        glm::vec4           q = f->getQuad_(i,0,0);    // 
        std::cout << i << " " << glm::to_string(q) << std::endl ;        
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
    unsigned char eps = 0 ; 
    assert( NPY<unsigned char>::compare(a,b,eps, true) == 0 ); 
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


void test_bitwiseOrUInt()
{
    NPY<float>* transforms0 = NPY<float>::make_identity_transforms(10); 
    unsigned num_transforms0 = transforms0->getNumItems(); 
    for(unsigned i=0 ; i < num_transforms0 ; i++)
    {
        unsigned transform_index = i + 1 ;   // 1-based
        transforms0->setUInt(i,0,3,0, transform_index ); 
    }

    unsigned geocode0 = 0xff ; 
    for(unsigned i=0 ; i < num_transforms0 ; i++)
    {
        transforms0->bitwiseOrUInt(i,0,3,0, (geocode0 << 24) );
    } 

    const char* path = "$TMP/NPY4Test/transforms.npy" ; 
    transforms0->save(path); 
    NPY<float>* transforms1 = NPY<float>::load(path);  
    unsigned num_transforms1 = transforms1->getNumItems(); 
    assert( num_transforms0 == num_transforms1 ); 

    for(unsigned i=0 ; i < num_transforms1 ; ++i)
    {
        unsigned packed = transforms1->getUInt(i,0,3,0);
        unsigned transform_index = packed & 0xffffff ; 
        unsigned geocode1 = packed >> 24 ; 

        LOG(info) << " transform_index " << transform_index << " geocode " << geocode1 ; 
        assert( transform_index == i + 1); 
        assert( geocode1 == geocode0 ); 
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
    test_getQuad_2(); 
    //test_setQuad_(); 
    //test_getQuad_crossType(); 
    //test_getQuad_crossType_cast_FAILS_RESHAPE(); 

    //test_bitwiseOrUInt(); 

    return 0 ; 
}

// om-;TEST=NPY4Test om-t

