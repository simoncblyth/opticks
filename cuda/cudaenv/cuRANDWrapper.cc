#include "cuRANDWrapper.hh"
#include "cuRANDWrapper_kernel.hh"
#include "LaunchSequence.hh"


void cuRANDWrapper::create_rng()
{
    m_dev_rng_states = create_rng_wrapper(m_launchseq);
}

void cuRANDWrapper::init_rng()
{
    init_rng_wrapper(
        m_launchseq,
        m_dev_rng_states, 
        m_seed, 
        m_offset
    );
}

void cuRANDWrapper::test_rng()
{

    unsigned int items = m_launchseq->getItems();

    m_test  = (float*)malloc(items*sizeof(float));

    test_rng_wrapper(
        m_launchseq, 
        m_dev_rng_states, 
        m_test
    );

    for(unsigned int i=0 ; i<items ; ++i)
    {
        if( i % m_imod == 0 )
        printf("%7u %10.4f \n", i, m_test[i] );
    } 

    free(m_test);

}


