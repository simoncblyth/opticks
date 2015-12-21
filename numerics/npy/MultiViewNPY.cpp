
#include "MultiViewNPY.hpp"
#include "ViewNPY.hpp"
#include "stdio.h"
#include "string.h"
#include "assert.h"



void MultiViewNPY::add(ViewNPY* vec)
{ 
    if(m_vecs.size() > 0)
    {
        ViewNPY* prior = m_vecs.back();
        assert(prior->getNPY() == vec->getNPY() && "LIMITATION : all ViewNPY in a MultiViewNPY must be views of the same underlying NPY");
    }
    m_vecs.push_back(vec);
}

unsigned int  MultiViewNPY::getNumVecs()
{ 
    return m_vecs.size();
}

ViewNPY* MultiViewNPY::operator [](const char* name)
{
    return find(name);
}


ViewNPY* MultiViewNPY::operator [](unsigned int index)
{
    return index < m_vecs.size() ? m_vecs[index] : NULL ;
}


ViewNPY* MultiViewNPY::find(const char* name)
{
    for(unsigned int i=0 ; i < m_vecs.size() ; i++)
    {
        ViewNPY* vnpy = m_vecs[i];
        if(strcmp(name, vnpy->getName())==0) return vnpy ;
    }
    return NULL ; 
}

void MultiViewNPY::Print(const char* msg)
{
    for(unsigned int i=0 ; i < m_vecs.size() ; i++)
    {
        ViewNPY* vnpy = m_vecs[i];
        vnpy->Print(msg);
    }
}

void MultiViewNPY::Summary(const char* msg)
{
    printf("%s\n", msg);
    for(unsigned int i=0 ; i < m_vecs.size() ; i++)
    {
        ViewNPY* vnpy = m_vecs[i];
        vnpy->Summary(msg);
    }
}
