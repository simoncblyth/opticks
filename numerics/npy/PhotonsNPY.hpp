#pragma once

class NPY ;

class PhotonsNPY {
   public:  
       PhotonsNPY(NPY* npy); // weak reference to NPY* only
       NPY* getNPY();
       void classify();

   public:  
       void dump(const char* msg);

  private:
        NPY*     m_npy ; 
 
};



inline PhotonsNPY::PhotonsNPY(NPY* npy) 
       :  
       m_npy(npy)
{
}

inline NPY* PhotonsNPY::getNPY()
{
    return m_npy ; 
}

