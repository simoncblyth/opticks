#pragma once

class TAry {
   public:
       TAry( void* dptr, unsigned int size, unsigned int num_bytes, void* hostcopy); 
   public:
       void transform(); 
       void copyToHost( void* host );
   private:
       void init();
   private:
       void*        m_dptr ; 
       unsigned int m_size ; 
       unsigned int m_num_bytes ; 
       void*        m_hostcopy ; 

};


inline TAry::TAry(void* dptr, unsigned int size, unsigned int num_bytes, void* hostcopy) :
    m_dptr(dptr),
    m_size(size),
    m_num_bytes(num_bytes),
    m_hostcopy(hostcopy)
{
}





