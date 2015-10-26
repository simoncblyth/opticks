#pragma once

template <typename T> class NPY ;


class MaterialLibNPY {
   public:  
       MaterialLibNPY(NPY<float>* mlib); 
   public:  
       void dump(const char* msg="MaterialLibNPY::dump");
       void dumpMaterial(unsigned int i);
   private:
       NPY<float>*   m_lib ; 
 
};

inline MaterialLibNPY::MaterialLibNPY(NPY<float>* mlib) 
       :  
       m_lib(mlib)
{
}





 
