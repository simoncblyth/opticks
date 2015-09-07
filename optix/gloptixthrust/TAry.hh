#pragma once

class TAry {
   public:
       TAry( void* dptr, unsigned int size ); 
   public:
       void transform(); 
   private:
       void init();
   private:
       void*         m_dptr ; 
       unsigned int m_size ; 

};


inline TAry::TAry(void* dptr, unsigned int size) :
    m_dptr(dptr),
    m_size(size)
{
}





