
#include "NPY.hpp"
#include "MaterialLibNPY.hpp"
#include "PLOG.hh"

MaterialLibNPY::MaterialLibNPY(NPY<float>* mlib) 
       :  
       m_lib(mlib)
{
}


void MaterialLibNPY::dump(const char* msg)
{
    unsigned int ni = m_lib->m_ni ;
    unsigned int nj = m_lib->m_nj ;
    unsigned int nk = m_lib->m_nk ;

    LOG(info) << msg 
              << " ni " << ni 
              << " nj " << nj 
              << " nk " << nk 
              ; 

    assert( nj == 39 && nk == 4 );

    for(unsigned int i=0 ; i<ni ; i++ )
    {
        dumpMaterial(i);
    }
}

void MaterialLibNPY::dumpMaterial(unsigned int)
{
    

}




