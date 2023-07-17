/**
QPMT_Test.cc : standalone built variant of om built QPMTTest.cc
=================================================================
**/

#include "QPMT.hh"

template<typename T>
struct QPMT_Test
{
    const QPMT<T>* qpmt ; 

    NP* energy_eV_domain ; 
    NP* lpmtcat_rindex ; 
    NP* lpmtcat_qeshape ; 

    QPMT_Test(const NPFold* jpmt); 

    void rindex_test(); 
    void qeshape_test(); 

    NPFold* serialize() const ; 
    void save() const ; 
};

template<typename T>
QPMT_Test<T>::QPMT_Test(const NPFold* jpmt  )
    :
    qpmt(new QPMT<T>(jpmt)),
    energy_eV_domain(NP::Linspace<T>( 1.55, 15.50, 1550-155+1 )), //  np.linspace( 1.55, 15.50, 1550-155+1 )  
    lpmtcat_rindex(nullptr),
    lpmtcat_qeshape(nullptr)
{
}

template<typename T>
void QPMT_Test<T>::rindex_test()
{
    lpmtcat_rindex = qpmt->lpmtcat_(qpmt_RINDEX,  energy_eV_domain) ; 

}
template<typename T>
void QPMT_Test<T>::qeshape_test()
{
    lpmtcat_qeshape = qpmt->lpmtcat_(qpmt_QESHAPE,  energy_eV_domain) ; 
}


template<typename T>
NPFold* QPMT_Test<T>::serialize() const
{
    NPFold* fold = new NPFold ; 
    fold->add_subfold("qpmt", qpmt->serialize() );  
    fold->add("energy_eV_domain", energy_eV_domain ); 
    fold->add("lpmtcat_rindex", lpmtcat_rindex ); 
    fold->add("lpmtcat_qeshape", lpmtcat_qeshape ); 
    return fold ; 
}


template<typename T>
void QPMT_Test<T>::save() const 
{
    NPFold* fold = serialize() ; 
    fold->save("$FOLD"); 
}


#include <cuda_runtime.h>
#include "OPTICKS_LOG.hh"
#include "get_jpmt_fold.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const NPFold* jpmt = get_jpmt_fold(); 

    QPMT_Test<float> t(jpmt); 

    t.rindex_test(); 
    t.qeshape_test(); 
 
    cudaDeviceSynchronize();
    t.save();  

    std::cout << t.qpmt->desc() << std::endl ; 

    return 0 ; 
}
