#include <cassert>

#include "G4Material.hh"
#include "G4MaterialTable.hh"

#include "X4MaterialTable.hh"
#include "X4Material.hh"

#include "GMaterial.hh"
#include "GMaterialLib.hh"
#include "Opticks.hh"


GMaterialLib* X4MaterialTable::Convert(GMaterialLib* mlib)
{
    const G4MaterialTable* mtab  = G4Material::GetMaterialTable();
    return Convert(mtab, mlib);
} 

GMaterialLib* X4MaterialTable::Convert(const G4MaterialTable* mtab, GMaterialLib* mlib) 
{
    X4MaterialTable xmt(mtab, mlib) ; 
    GMaterialLib* mlib_ = xmt.getMaterialLib();
    if(mlib) 
    {
       assert( mlib == mlib_ ); 
    }
    return mlib_ ; 
}

GMaterialLib* X4MaterialTable::getMaterialLib()
{
    return m_mlib ;
}

X4MaterialTable::X4MaterialTable(const G4MaterialTable* mtab, GMaterialLib* mlib)
    :
    m_mtab(mtab),
    m_ok(Opticks::GetOpticks()),
    m_mlib(mlib == NULL ? new GMaterialLib(m_ok) : mlib)
{
    init();
}

void X4MaterialTable::init()
{
    unsigned nmat = G4Material::GetNumberOfMaterials();
    assert( nmat == m_mtab->size() ) ; 
  
    for(unsigned i=0 ; i < nmat ; i++)
    {   
        G4Material* material = (*m_mtab)[i];

        GMaterial* mat = X4Material::Convert( material ); 

        mat->setIndex(i) ; 

        m_mlib->add(mat) ; // creates standardized material
    }
}



