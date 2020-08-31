#include "NPY.hpp"
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




int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    //test_make_modulo_selection();  
    //test_write_item_(); 
    //test_write_item_big(); 

    test_concat<unsigned char>(); 

    return 0 ; 
}


