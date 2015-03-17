#include "cuRANDWrapper.hh"
#include "cuRANDWrapper_kernel.hh"
#include "LaunchSequence.hh"


void cuRANDWrapper::create_rng()
{
    m_dev_rng_states = create_rng_wrapper(m_launchseq);
}

void cuRANDWrapper::init_rng(const char* tag)
{
    LaunchSequence* seq = m_launchseq->copy() ;
    seq->setTag(tag);

    init_rng_wrapper(
        seq,
        m_dev_rng_states, 
        m_seed, 
        m_offset
    );

    m_launchrec.push_back(seq); 
}

void cuRANDWrapper::test_rng(const char* tag)
{
    LaunchSequence* seq = m_launchseq->copy() ;
    seq->setTag(tag);

    unsigned int items = seq->getItems();

    m_test  = (float*)malloc(items*sizeof(float));

    test_rng_wrapper(
        seq, 
        m_dev_rng_states, 
        m_test
    );

    for(unsigned int i=0 ; i<items ; ++i)
    {
        if( i % m_imod == 0 )
        printf("%7u %10.4f \n", i, m_test[i] );
    } 

    free(m_test);

    m_launchrec.push_back(seq); 
}


void cuRANDWrapper::Summary(const char* msg)
{
    unsigned int nrec = m_launchrec.size();
    for(unsigned int i=0 ; i < nrec ; i++)
    {
        LaunchSequence* seq = m_launchrec[i];  
        seq->Summary(msg); 
    }
}


