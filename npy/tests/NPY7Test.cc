#include "NPY.hpp"
#include "SSys.hh"
#include "OPTICKS_LOG.hh"

NPY<float>* test_make()
{
    std::vector<glm::mat4> mats ; 
    glm::mat4 m(1.f); 
    for(unsigned i=0 ; i < 10 ; i++) mats.push_back(m) ;
    NPY<float>* a = NPY<float>::make(mats.size(), 4, 4);
    a->read( mats.data() ); 

    const char* path = "$TMP/NPY7Test/mats.npy" ;
    LOG(info) << "saving " << path ; 
    a->save(path); 
    SSys::npdump(path, "np.float32"); 

    return a ; 
}


void test_save_nulldir(const NPY<float>* a )
{
    a->save(nullptr, "test_save_nulldir.npy" );   // fails 
    //a->save(".", "test_save_nulldir.npy" );     // works
    //a->save("test_save_nulldir.npy" );          // fails
}


void test_setAllValue()
{
   LOG(info) ; 

   NPY<float>* ip = NPY<float>::load("$HOME/.opticks/InputPhotons/InwardsCubeCorners1.npy"); 
   if( ip == nullptr ) return ; 
 
   float wavelength_nm = 444.f ; 
   int j = 2 ; 
   int k = 3 ; 
   int l = 0 ; 
   ip->setAllValue(j, k, l, wavelength_nm );  
   ip->dump(); 

   ip->save("$TMP/test_setAllValue.npy"); 
}


void test_MakeFloat()
{
    unsigned ni = 10 ; 
    unsigned nj = 4  ; 
    unsigned nk = 2  ; 
    unsigned nl = 2  ; 
    unsigned nm = 4  ; 

    NPY<double>* src = NPY<double>::make( ni, nj, nk, nl, nm ); 
    src->fillIndexFlat(); 
    src->dump("src"); 

    NPY<float>* dst = NPY<double>::MakeFloat(src); 
    dst->dump("dst"); 

    for(unsigned i=0 ; i < ni ; i++){
    for(unsigned j=0 ; j < nj ; j++){
    for(unsigned k=0 ; k < nk ; k++){
    for(unsigned l=0 ; l < nl ; l++){
    for(unsigned m=0 ; m < nm ; m++){

        double s = src->getValue(i,j,k,l,m) ;    
        float  d = dst->getValue(i,j,k,l,m) ;    
        assert( float(s) == d ); 

    }
    }
    }
    }
    }
}





void test_interp()
{
    unsigned na = 3 ; 
    NPY<float>* a = NPY<float>::make(na,2); 
    a->zero(); 
    for(unsigned i=0 ; i < na ; i++)
    {
        a->setValue(i,0,0,0,  float(i)) ; 
        a->setValue(i,1,0,0,  10.f*float(i+1)) ;  
    }

    unsigned nb = 100 ; 
    NPY<float>* b = NPY<float>::make(nb,2); 
    b->zero(); 
 
    for(unsigned i=0 ; i < nb ; i++)
    {
        float x = -0.5f + i*0.1f ; 
        float y = a->interp(x); 

        std::cout  
            << std::setw(3) << i 
            << " x " << std::setw(10) << std::setprecision(5) << std::fixed << x 
            << " y " << std::setw(10) << std::setprecision(5) << std::fixed << y 
            << std::endl 
            ;

        b->setValue(i,0,0,0,  x) ; 
        b->setValue(i,1,0,0,  y) ;  
    }
    const char* FOLD = "$TMP/NPY7Test/test_interp" ; 

    a->save(FOLD, "src.npy"); 
    b->save(FOLD, "dst.npy"); 

    LOG(info) << " saved to " << FOLD ; 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    //NPY<float>* a = test_make(); 
    //test_save_nulldir(a); 

    //test_setAllValue(); 

    //test_MakeFloat(); 
    test_interp(); 

    return 0 ; 
}

