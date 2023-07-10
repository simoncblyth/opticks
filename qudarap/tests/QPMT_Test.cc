/**
QPMT_Test.cc 
=================
**/

#include "QPMT.hh"

template<typename T>
struct QPMT_Test
{
    const QPMT<T>& qpmt ; 

    NP* energy_eV_domain ; 
    NP* lpmtcat_rindex ; 
    NP* lpmtcat_qeshape ; 

    QPMT_Test(const QPMT<T>& qpmt ); 

    void rindex_test(); 
    void qeshape_test(); 

    NPFold* make_fold() const ; 
    void save() const ; 
};

template<typename T>
QPMT_Test<T>::QPMT_Test(const QPMT<T>& qpmt_ )
    :
    qpmt(qpmt_),
    energy_eV_domain(NP::Linspace<T>( 1.55, 15.50, 1550-155+1 )), //  np.linspace( 1.55, 15.50, 1550-155+1 )  
    lpmtcat_rindex(nullptr),
    lpmtcat_qeshape(nullptr)
{
}

template<typename T>
void QPMT_Test<T>::rindex_test()
{
    lpmtcat_rindex = qpmt.lpmtcat_(qpmt_RINDEX,  energy_eV_domain) ; 

}
template<typename T>
void QPMT_Test<T>::qeshape_test()
{
    lpmtcat_qeshape = qpmt.lpmtcat_(qpmt_QESHAPE,  energy_eV_domain) ; 
}


template<typename T>
NPFold* QPMT_Test<T>::make_fold() const
{
    NPFold* fold = new NPFold ; 
    fold->add_subfold("qpmt", qpmt.serialize() );  
    fold->add("energy_eV_domain", energy_eV_domain ); 
    fold->add("lpmtcat_rindex", lpmtcat_rindex ); 
    fold->add("lpmtcat_qeshape", lpmtcat_qeshape ); 
    return fold ; 
}


template<typename T>
void QPMT_Test<T>::save() const 
{
    NPFold* fold = make_fold() ; 
    fold->save("$FOLD"); 
}


#include <cuda_runtime.h>

#ifdef WITH_JPMT
#include "JPMT.h"
#else
#include "SPMT.h"
#endif


#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    NPFold* fold = nullptr ; 

#ifdef WITH_JPMT
    std::cout << "QPMT_Test.cc : WITH_JPMT " << std::endl ; 
    JPMT pmt ; 
    std::cout << pmt.desc() << std::endl ;
    pmt.save("$FOLD") ; 

    fold = pmt.get_fold(); 
#else
    std::cout << "QPMT_Test.cc : NOT-WITH_JPMT " << std::endl ; 
    SPMT* pmt = SPMT::Load();
    if(pmt == nullptr)
    {
        std::cout << "QPMT_Test.cc FAILED TO SPMT::Load ? IS GEOM envvar defined ? " << std::endl ; 
        return 1 ; 
    }

    fold = pmt->get_fold(); 
#endif

    QPMT<float> qp(fold) ; 
    std::cout << qp.desc() << std::endl ; 

    QPMT_Test<float> t(qp); 
    t.rindex_test(); 
    t.qeshape_test(); 
 
    cudaDeviceSynchronize();
    t.save();  

    std::cout << qp.desc() << std::endl ; 

    return 0 ; 
}
