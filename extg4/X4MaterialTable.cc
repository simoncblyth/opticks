#include <cassert>

#include "G4Material.hh"
#include "G4MaterialTable.hh"

#include "X4MaterialTable.hh"
#include "X4Material.hh"

#include "GMaterial.hh"
#include "GMaterialLib.hh"
#include "Opticks.hh"

#include "PLOG.hh"


G4Material* X4MaterialTable::Get(unsigned idx)
{
    unsigned nmat = G4Material::GetNumberOfMaterials();
    assert( idx < nmat );
    G4MaterialTable* mtab = G4Material::GetMaterialTable();
    G4Material* material = (*mtab)[idx];
    assert( material->GetIndex() == idx );
    return material ; 
}



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
    LOG(m_level) << ". G4 nmat " << nmat ;  
    for(unsigned i=0 ; i < nmat ; i++)
    {   
        G4Material* material = Get(i) ; 
        G4MaterialPropertiesTable* mpt = material->GetMaterialPropertiesTable();

        if( mpt == NULL )
        {
            LOG(warning) << "skip convert of material with no mpt " << material->GetName() ; 
            continue ;  
        }

        GMaterial* mat = X4Material::Convert( material ); 

        assert( mat->getIndex() == i ); // this is not the lib, no danger of triggering a close

        m_mlib->add(mat) ;    // creates standardized material
        m_mlib->addRaw(mat) ; // stores as-is
    }
}



