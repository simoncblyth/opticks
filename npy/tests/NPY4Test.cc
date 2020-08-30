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




int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    //test_make_modulo_selection();  
    test_write_item_(); 

    return 0 ; 
}


