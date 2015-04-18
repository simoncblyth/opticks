
#include "MultiVecNPY.hpp"
#include "VecNPY.hpp"
#include "stdio.h"
#include "string.h"
#include "assert.h"


MultiVecNPY::MultiVecNPY()
{
}

void MultiVecNPY::add(VecNPY* vec)
{ 
    if(m_vecs.size() > 0)
    {
        VecNPY* prior = m_vecs.back();
        assert(prior->getNPY() == vec->getNPY() && "LIMITATION : all VecNPY in a MultiVecNPY must be views of the same underlying NPY");        
    }
    m_vecs.push_back(vec);
}

unsigned int  MultiVecNPY::getNumVecs()
{ 
    return m_vecs.size();
}

VecNPY* MultiVecNPY::operator [](const char* name)
{
    return find(name);
}

VecNPY* MultiVecNPY::operator [](unsigned int index)
{
    return index < m_vecs.size() ? m_vecs[index] : NULL ;
}


VecNPY* MultiVecNPY::find(const char* name)
{
    for(unsigned int i=0 ; i < m_vecs.size() ; i++)
    {
        VecNPY* vnpy = m_vecs[i];
        if(strcmp(name, vnpy->getName())==0) return vnpy ;
    }
    return NULL ; 
}

void MultiVecNPY::Print(const char* msg)
{
    for(unsigned int i=0 ; i < m_vecs.size() ; i++)
    {
        VecNPY* vnpy = m_vecs[i];
        vnpy->Print(msg);
    }
}

void MultiVecNPY::Summary(const char* msg)
{
    printf("%s\n", msg);
    for(unsigned int i=0 ; i < m_vecs.size() ; i++)
    {
        VecNPY* vnpy = m_vecs[i];
        vnpy->Summary(msg);
    }
}
