

// bregex-
#include "dbg.hh"

#include "NPY.hpp"
#include "BLog.hh"
#include "G4StepNPY.hpp"

#include <vector>
#include <iostream>
#include <cassert>

#include <glm/glm.hpp>

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
   aa->save("/tmp/aa.npy"); 
   bb->save("/tmp/bb.npy"); 
   bb->reshape(-1, n, 1, 4);
   bb->save("/tmp/cc.npy"); 
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
   NPY<unsigned int>* solidBuf = NPY<unsigned int>::load("/tmp/solidBuf.npy");
   solidBuf->dump();
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
   a->save("/tmp/test_empty_add.npy");
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

   dom->save("/tmp/test_add.npy");

}

void test_slice()
{
   NPY<float>* dom = NPY<float>::make(10,4,4) ;
   dom->fill(0.f);
   dom->dump();   

   NPY<float>* som = dom->make_slice("0:3");
   som->dump();   
}


void test_setQuad()
{
   NPY<float>* dom = NPY<float>::make(10,1,4) ;
   dom->fill(0.f);

   glm::vec4 q0(0.f,1.f,2.f,3.f);
   glm::vec4 q1(1.f,1.f,2.f,3.f);
   glm::vec4 q9(9.f,1.f,2.f,3.f);

   dom->setQuad(q0, 0,0);
   dom->setQuad(q1, 1,0);
   dom->setQuad(q9, 9,0);
    
   dom->save("/tmp/test_setQuad.npy");
}


void test_ctor()
{
    std::vector<int> shape = {2,2} ;
    std::vector<float> data = {1.f,2.f,3.f,4.f}  ;
    std::string metadata = "{}";

    NPY<float> npy(shape,data,metadata) ;
    std::cout << npy.description("npy") << std::endl ; 
}

void test_path()
{
    std::string path = NPY<float>::path("cerenkov", "1", "dayabay");
    std::cout << path << std::endl ; 
}

void test_load()
{
    NPY<float>* npy = NPY<float>::load("cerenkov","1", "dayabay");
    std::cout << npy->description("npy") << std::endl ; 
}

void test_save_path()
{
    NPY<float>* npy = NPY<float>::load("cerenkov","1", "dayabay");
    std::cout << npy->description("npy") << std::endl ; 
    npy->save("/tmp/test_save_path.npy");
}



void test_load_path()
{
    const char* path = "/tmp/slowcomponent.npy" ;
    //const char* path = "/usr/local/env/cerenkov/1.npy" ;
    NPY<float>* npy = NPY<float>::debugload(path);
    if(npy) npy->Summary(path);
}


void test_load_missing()
{
    NPY<float>* npy = NPY<float>::load("cerenkov","missing", "dayabay");
    if(npy) std::cout << npy->description("npy") << std::endl ; 
}

void test_g4stepnpy()
{
    NPY<float>* npy = NPY<float>::load("cerenkov","1", "dayabay");
    G4StepNPY* step = new G4StepNPY(npy);   
    step->dump("G4StepNPY");
}

void test_make_modulo()
{
    NPY<float>* npy0 = NPY<float>::load("cerenkov","1", "dayabay");
    G4StepNPY* step0 = new G4StepNPY(npy0);   
    step0->dump("G4StepNPY0");


    NPY<float>* npy1 = NPY<float>::make_modulo(npy0, 10) ;
    G4StepNPY* step1 = new G4StepNPY(npy1);   
    step1->dump("G4StepNPY1");

}


void test_g4stepnpy_materials()
{

    //const char* det = "dayabay" ; 
    const char* det = "juno" ; 

    NPY<float>* npy = NPY<float>::load("cerenkov","1", det);
    G4StepNPY* step = new G4StepNPY(npy);   
    step->dump("G4StepNPY");

    std::set<int> s = npy->uniquei(0,2);
    typedef std::set<int>::const_iterator SII ; 

    for(SII it=s.begin() ; it != s.end() ; it++)
    {
        printf(" %d \n", *it ); 
    }


    std::map<int,int> m = npy->count_uniquei(0,2);
    typedef std::map<int, int>::const_iterator MII ; 

    for(MII it=m.begin() ; it != m.end() ; it++)
    {
        printf(" %d : %d \n", it->first, it->second ); 
    }



}






void test_getData()
{
    NPY<float>* npy = NPY<float>::load("cerenkov","1", "dayabay");
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
    NPY<float>* c1 = NPY<float>::load("cerenkov","1", "dayabay");
    NPY<float>* s1 = NPY<float>::load("scintillation","1", "dayabay");
    unsigned int n_c1 = c1->getUSum(0, 3);
    unsigned int n_s1 = s1->getUSum(0, 3);
    printf("test_getUSum n_c1:%u n_c1:%u tot:%u \n", n_c1, n_s1, n_c1+n_s1);
}

void test_string()
{
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
    s->save("/tmp/test_string.npy");

/*

   messes up curiously when go beyond 8 chars, expected just truncation 
   getting garbage

In [55]: np.load("/tmp/test_string.npy").view(np.dtype("S8"))
Out[55]: 
array([[['hello123']]], 
      dtype='|S8')

In [54]: np.load("/tmp/test_string.npy").view(np.dtype("S8"))
Out[54]: 
array([[['|u~lo123']]], 
      dtype='|S8')




*/


}


int main(int argc, char** argv )
{
    BLog nl(argc, argv);


    //test_ctor();
    //test_path();
    //test_load();
    //test_load_missing();
    //test_g4stepnpy();
    //test_getData();

    //test_getUSum();
    //test_load_path();
    //test_save_path();

    //test_setQuad();
    //test_string();

    //test_g4stepnpy();   
    //test_make_modulo();   
    //test_g4stepnpy_materials();

    //test_slice();

    //test_add();
    //test_empty_add();
    //test_dump();

    //test_transform();
    //test_reshape();


    NPYBase::setGlobalVerbose(true);

    test_repeat();

    DBG("NPYTest::", "main", "after repeat" );

    return 0 ;
}
