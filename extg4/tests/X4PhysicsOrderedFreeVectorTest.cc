#include "G4PhysicsOrderedFreeVector.hh"
#include "X4PhysicsOrderedFreeVector.hh"
#include "NPY.hpp"

#include "OPTICKS_LOG.hh"


const char* FOLD = "/tmp/X4PhysicsOrderedFreeVectorTest" ; 


void test_convert()
{
    size_t len = 10 ;  
    G4double* energy = new G4double[len] ;
    G4double* value = new G4double[len] ;
 
    for(int i=0 ; i < int(len) ; i++)
    {
        energy[i] = G4double(i)  ; 
        value[i] = 100.*G4double(i); 
    }

    G4PhysicsOrderedFreeVector* vec = new G4PhysicsOrderedFreeVector(energy, value, len);  
    X4PhysicsOrderedFreeVector* xvec = new X4PhysicsOrderedFreeVector(vec); 

    NPY<double>* d = xvec->convert<double>(); 
    d->dump(); 

    NPY<float>*  f = xvec->convert<float>(); 
    f->dump(); 
}


void test_Load0()
{
    const char* keydir = getenv("OPTICKS_KEYDIR"); 
    if( keydir == nullptr ) return ; 

    NPY<double>* a = NPY<double>::load(keydir, "GScintillatorLib/LS_ori/RINDEX.npy") ; 
    a->pscale(1e6, 0u); 
    a->pdump("test_Load0"); 
}


void VecDump(G4PhysicsOrderedFreeVector* vec)
{
    std::cout << "VecDump" << std::endl ; 
    G4cout << *vec << G4endl ;
    std::cout << "G4PhysicsOrderedFreeVector:: " << std::endl ;  
    std::cout 
        << std::setw(30) << "GetMinLowEdgeEnergy() " 
        << std::fixed << std::setw(10) << std::setprecision(5) << vec->GetMinLowEdgeEnergy() 
        << std::endl 
        << std::setw(30) << "GetMinValue() " 
        << std::fixed << std::setw(10) << std::setprecision(5) << vec->GetMinValue() 
        << std::endl 
        << std::setw(30) << "GetMaxLowEdgeEnergy() " 
        << std::fixed << std::setw(10) << std::setprecision(5) << vec->GetMaxLowEdgeEnergy() 
        << std::endl 
        << std::setw(30) << "GetMaxValue() " 
        << std::fixed << std::setw(10) << std::setprecision(5) << vec->GetMaxValue() 
        << std::endl 
        ; 

}


void test_Load1()
{
    const char* keydir = getenv("OPTICKS_KEYDIR"); 
    if( keydir == nullptr ) return ; 

    double en_scale = 1e6 ; 
    X4PhysicsOrderedFreeVector* xvec = X4PhysicsOrderedFreeVector::Load(keydir, "GScintillatorLib/LS_ori/RINDEX.npy", en_scale );     
    VecDump(xvec->vec); 
}

void test_Value()
{

    const char* keydir = getenv("OPTICKS_KEYDIR"); 
    if( keydir == nullptr ) return ; 

    double en_scale = 1e6 ; 

    X4PhysicsOrderedFreeVector* xvec = X4PhysicsOrderedFreeVector::Load(keydir, "GScintillatorLib/LS_ori/RINDEX.npy", en_scale );     
    G4PhysicsOrderedFreeVector* vec = xvec->vec ; 
    const NPY<double>* src = xvec->src ; 
    VecDump(vec); 

    src->save(FOLD, "src.npy"); 

    double el = 0. ; 
    double eh = 16. ; 

    unsigned nb = 1000 ; 
    NPY<double>* dst = NPY<double>::make(nb, 3 ); 
    dst->zero(); 
    for(unsigned i=0 ; i < nb ; i++ )  
    {
        double e = el + (eh-el)*double(i)/double(nb-1) ; 
        double v0 = vec->Value(e); 
        double v1 = src->interp(e); 
        double dv = std::abs(v1 - v0) ;  
        std::cout 
            << " i " << std::setw(3) << i 
            << " e " << std::setw(10) << std::setprecision(5) << std::fixed << e 
            << " v0 " << std::setw(10) << std::setprecision(5) << std::fixed << v0
            << " v1 " << std::setw(10) << std::setprecision(5) << std::fixed << v1
            << " dv " << std::setw(10) << std::setprecision(5) << std::fixed << dv
            << " dv*1e9 " << std::setw(10) << std::setprecision(5) << std::fixed << dv*1e9
            << std::endl 
            ; 

        dst->setValue(i,0,0,0,   e ); 
        dst->setValue(i,1,0,0,   v0 ); 
        dst->setValue(i,2,0,0,   v1 ); 

        assert( dv == 0. ); 

    }


    dst->save(FOLD, "dst.npy"); 

}





int main(int argc, char** argv)
{  
    //test_convert(); 
    //test_Load0();  
    //test_Load1();  
    test_Value();  
    return 0 ; 
}
