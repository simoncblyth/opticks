
#include "NPY.hpp"
#include "OpEvt.hh"


OpEvt::OpEvt() 
    :
    m_genstep(NULL)
{
}

void OpEvt::addGenstep( float* data, unsigned num_float )
{
    assert( num_float == 6*4 ) ;     
    if(!m_genstep) m_genstep = NPY<float>::make(0,6,4) ; 
    m_genstep->add(data, num_float ); 
}

unsigned OpEvt::getNumGensteps() const 
{
    return m_genstep ? m_genstep->getShape(0) : 0 ; 
}

NPY<float>* OpEvt::getEmbeddedGensteps()
{
    return m_genstep ; 
}



void OpEvt::loadEmbeddedGensteps(const char* path)
{
    m_genstep = NPY<float>::load(path) ; 
}

void OpEvt::saveEmbeddedGensteps(const char* path) const 
{
    if(!m_genstep) return ; 
    m_genstep->save(path) ; 
}


void OpEvt::resetGensteps() 
{
    m_genstep->reset();
    assert( getNumGensteps() == 0 );
}




