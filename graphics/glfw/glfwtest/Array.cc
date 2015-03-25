#include "Array.hh"

Array::Array(unsigned int length, const float* values)
   : 
   m_length(length),
   m_values(values)
{
}

Array::~Array()
{
}

unsigned int Array::getLength()
{
   return m_length ;
}

const float* Array::getValues()
{
   return m_values ;
}



