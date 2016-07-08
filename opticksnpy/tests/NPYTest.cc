#include "NPY_FLAGS.hh"


#include <vector>
#include <iostream>
#include <cassert>


#include "BBufSpec.hh"

// npy-
#include "NGLM.hpp"
#include "NPY.hpp"

#include "BRAP_LOG.hh"
#include "PLOG.hh"

void test_repeat()
{
   // see associated tests/NPYTest.py 

   NPY<int>* aa = NPY<int>::make(100,1,4) ;
   aa->zero();

   for(unsigned int i=0 ; i < aa->getShape(0) ; i++){
   for(unsigned int j=0 ; j < aa->getShape(1) ; j++){
   for(unsigned int k=0 ; k < aa->getShape(2) ; k++){
   for(unsigned int l=0 ; l < std::max(1u,aa->getShape(3)) ; l++)
   {
       int value = i*100+j*10+k ;
       aa->setValue(i,j,k,l, value);  
   }
   }
   }
   }

   unsigned int n = 10 ; 
   NPY<int>* bb = NPY<int>::make_repeat(aa, n) ; 
   aa->save("$TMP/aa.npy"); 
   bb->save("$TMP/bb.npy"); 
   bb->reshape(-1, n, 1, 4);
   bb->save("$TMP/cc.npy"); 
}


void test_reshape()
{
   NPY<int>* rx = NPY<int>::make(100,10,2,4) ;
   rx->zero();

   rx->Summary("before reshape");

   unsigned int ni = rx->getShape(0); 
   unsigned int nj = rx->getShape(1); 
   unsigned int nk = rx->getShape(2); 
   unsigned int nl = rx->getShape(3); 

   rx->reshape( ni*nj, nk, nl, 0 );

   rx->Summary("after reshape");

}



void test_transform()
{
   NPY<float>* dom = NPY<float>::make(10,1,4) ;
   dom->fill(0.f);

   glm::vec4 q0(0.f,1.f,2.f,1.f);
   glm::vec4 q1(1.f,1.f,2.f,1.f);
   glm::vec4 q9(9.f,1.f,2.f,1.f);

   dom->setQuad(q0, 0,0);
   dom->setQuad(q1, 1,0);
   dom->setQuad(q9, 9,0);

   dom->dump();
    
   //glm::mat4 m(glm::translate(glm::vec3(10.,10.,10.)));
   //glm::mat4 m = glm::translate(glm::vec3(10.,10.,10.));

   glm::vec3 tr(10.,10.,10);
   glm::vec3 sc(10.,5.,-10);

   glm::mat4 m = glm::scale(glm::translate(glm::mat4(1.0), tr), sc);

   NPY<float>* tdom = dom->transform(m);
   tdom->dump(); 
}


void test_dump()
{
   const char* path = "$TMP/dom.npy";  
   typedef unsigned int T ; 

   NPY<T>* a = NPY<T>::make(10,1,4) ;
   a->fill(0);
   a->save(path);

   NPY<T>* b = NPY<T>::load(path);
   b->dump();
}


void test_empty_add()
{
   std::vector<int> shape ; 
   shape.push_back(0); 
   shape.push_back(4); 
   shape.push_back(4); 

   NPY<float>* a = NPY<float>::make(shape);
   a->zero();

   for(unsigned int i=0 ; i < 10 ; i++)
   {
       NPY<float>* ext = NPY<float>::make(1,4,4) ;
       ext->fill(float(i));
       a->add(ext);
   }

   a->dump();
   a->save("$TMP/test_empty_add.npy");
}


void test_add()
{
   NPY<float>* dom = NPY<float>::make(10,4,4) ;
   dom->fill(1.f);
   dom->dump();   

   NPY<float>* ext = NPY<float>::make(3,4,4) ;
   ext->fill(2.f);
   ext->dump();   

   dom->add(ext);
   dom->dump();

   dom->save("$TMP/test_add.npy");

}


void test_slice()
{
   LOG(info) << "test_slice" ; 
   NPY<float>* dom = NPY<float>::make(10,4,4) ;
   dom->fill(0.f);
   dom->dump();   

   NPY<float>* som = dom->make_slice("0:3");
   som->dump();   
   LOG(info) << "test_slice DONE" ; 
}


void test_setQuad()
{
   LOG(info) << "test_setQuad" ; 
   NPY<float>* dom = NPY<float>::make(10,1,4) ;
   dom->fill(0.f);

   glm::vec4 q0(0.f,1.f,2.f,3.f);
   glm::vec4 q1(1.f,1.f,2.f,3.f);
   glm::vec4 q9(9.f,1.f,2.f,3.f);

   dom->setQuad(q0, 0,0);
   dom->setQuad(q1, 1,0);
   dom->setQuad(q9, 9,0);
    
   dom->save("$TMP/test_setQuad.npy");
}





void test_ctor_segfaults()
{
    LOG(info) << "test_ctor" ; 
    std::vector<int> shape ;
    shape.push_back(1);
    shape.push_back(1);
    shape.push_back(4);

    std::vector<float> data ;
    data.push_back( 1.f);
    data.push_back( 2.f);
    data.push_back( 3.f);
    data.push_back( 4.f);

    std::string metadata = "{}";

    NPY<float>* npy = new NPY<float>(shape,data,metadata) ;
    LOG(info) << "test_ctor (1)" ; 
    std::cout << npy->description("npy") << std::endl ; 
    LOG(info) << "test_ctor DONE" ; 
}

void test_path()
{
    LOG(info) << "test_path" ; 
    std::string path = NPY<float>::path("cerenkov", "1", "dayabay");
    LOG(info) << "test_path path:" << path ;
    LOG(info) << "test_path DONE" ; 
}

void test_load()
{
    LOG(info) << "test_load" ; 
    NPY<float>* npy = NPY<float>::load("cerenkov","1", "dayabay");
    if(npy) std::cout << npy->description("npy") << std::endl ; 
}

void test_save_path()
{
    LOG(info) << "test_save_path" ; 
    NPY<float>* npy = NPY<float>::load("cerenkov","1", "dayabay");
    if(npy)
    { 
       std::cout << npy->description("npy") << std::endl ; 
       npy->save("$TMP/test_save_path.npy");
    }
}


void test_load_path_throws()
{
    LOG(info) << "test_load_path (throws std::runtime_error causing abort as not caught)" ; 

    const char* path = "$TMP/slowcomponent.npy" ;
    //const char* path = "/usr/local/env/cerenkov/1.npy" ;
    NPY<float>* npy = NPY<float>::debugload(path);
    if(npy) npy->Summary(path);
}


void test_load_missing()
{
    LOG(info) << "test_load_missing" ; 
    NPY<float>* npy = NPY<float>::load("cerenkov","missing", "dayabay");
    if(npy) std::cout << npy->description("npy") << std::endl ; 
}




void test_getData()
{
    LOG(info) << "test_getData" ; 
    NPY<float>* npy = NPY<float>::load("cerenkov","1", "dayabay");
    if(!npy) return ; 

    float* data = npy->getValues();

    for(unsigned int i=0 ; i < 16 ; i++ )
    {
        uif_t uif ;
        uif.f = *(data+i) ;
        printf(" %3u : %15f f   %15d i  %15u u  \n", i, uif.f, uif.i, uif.u );
    }

    char* raw = (char*)data ;
    for(unsigned int i=0 ; i < 16 ; i++ )
    {
        char c = *(raw+i) ;
        printf(" %3u : %d i   %x x \n", i, c, c);
    }
    std::cout << npy->description("npy") << std::endl ; 
}

void test_getUSum()
{
    LOG(info) << "test_getUSum" ; 
    NPY<float>* c1 = NPY<float>::load("cerenkov","1", "dayabay");
    NPY<float>* s1 = NPY<float>::load("scintillation","1", "dayabay");
    unsigned int n_c1 = c1 ? c1->getUSum(0, 3) : 0 ;
    unsigned int n_s1 = s1 ? s1->getUSum(0, 3) : 0 ;
    printf("test_getUSum n_c1:%u n_c1:%u tot:%u \n", n_c1, n_s1, n_c1+n_s1);
}

void test_string()
{
    LOG(info) << "test_string" ; 
    typedef unsigned long long ULL ; 

    std::cout << "sizeof(ULL) " << sizeof(ULL) << std::endl ; 

    assert(sizeof(ULL) == 8 );

    ULL* vals = new ULL[1] ;  
    const char* msg = "hello123" ;

    vals[0] = 0 ;
    char* c = (char*)msg ; 
    unsigned int i(0) ; 
    while(*c)
    {
        printf(" i %u c %c \n", i, *c );
        ULL ull = *c ; 


        vals[0] |= (ull & 0xFF) << (i*8) ; 

        i++ ;     
        c++ ; 
    }     

    NPY<ULL>* s = NPY<ULL>::make(1, 1, 1);
    s->setData(vals);
    s->save("$TMP/test_string.npy");

/*

   messes up curiously when go beyond 8 chars, expected just truncation 
   getting garbage

In [55]: np.load("$TMP/test_string.npy").view(np.dtype("S8"))
Out[55]: 
array([[['hello123']]], 
      dtype='|S8')

In [54]: np.load("$TMP/test_string.npy").view(np.dtype("S8"))
Out[54]: 
array([[['|u~lo123']]], 
      dtype='|S8')


*/


}



void test_getBufSpec()
{
   NPY<float>* buf = NPY<float>::make(1,1,4) ;
   buf->fill(1.f);
   buf->dump();   

   BBufSpec* bs = buf->getBufSpec();

   bs->Summary("test_getBufSpec");

   assert(bs->id == -1);
   assert(bs->ptr != NULL);
   assert(bs->num_bytes > 0);
   assert(bs->target == -1);

}




int main(int argc, char** argv )
{
    PLOG_(argc, argv);
    BRAP_LOG_ ;   

    //BRAP_LOG::Initialize(plog::get(), plog::get()->getMaxSeverity() );


    NPYBase::setGlobalVerbose(true);

    test_repeat();
    test_reshape();
    test_transform();
    test_dump();
    test_empty_add();
    test_add();
    test_slice();
    test_setQuad();

    test_ctor_segfaults();

    test_path();
    test_load();
    test_save_path();

    //test_load_path_throws();

    test_load_missing();

    test_getData();
    test_getUSum();
    test_string();

    test_getBufSpec();

    return 0 ;
}
