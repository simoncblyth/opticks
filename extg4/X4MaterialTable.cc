#include <cassert>

#include "G4Material.hh"
#include "G4MaterialTable.hh"

#include "X4MaterialTable.hh"
#include "X4Material.hh"

#include "GMaterial.hh"
#include "GMaterialLib.hh"
#include "Opticks.hh"


void X4MaterialTable::Convert(GMaterialLib* mlib)
{
    assert( mlib->getNumMaterials() == 0 ); 
    X4MaterialTable xmt(mlib) ; 
    assert( mlib == xmt.getMaterialLib() );
}

GMaterialLib* X4MaterialTable::getMaterialLib()
{
    return m_mlib ;
}

X4MaterialTable::X4MaterialTable(GMaterialLib* mlib)
    :
    m_mtab(G4Material::GetMaterialTable()),
    m_mlib(mlib)
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
  
        assert( material->GetIndex() == i );

        GMaterial* mat = X4Material::Convert( material ); 

        mat->setIndex(i) ; 

        m_mlib->add(mat) ; // creates standardized material
    }
}



