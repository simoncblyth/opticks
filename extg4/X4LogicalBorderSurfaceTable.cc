
#include "G4LogicalBorderSurface.hh"

#include "X4LogicalBorderSurfaceTable.hh"
#include "X4LogicalBorderSurface.hh"

#include "GBorderSurface.hh"
#include "GSurfaceLib.hh"

#include "PLOG.hh"


void X4LogicalBorderSurfaceTable::Convert( GSurfaceLib* slib )
{
    X4LogicalBorderSurfaceTable xtab(slib); 
}

X4LogicalBorderSurfaceTable::X4LogicalBorderSurfaceTable(GSurfaceLib* slib )
    :
    m_table(G4LogicalBorderSurface::GetSurfaceTable()),
    m_slib(slib)
{
    init();
}


void X4LogicalBorderSurfaceTable::init()
{
    unsigned num_lbs = G4LogicalBorderSurface::GetNumberOfBorderSurfaces() ; 
    assert( num_lbs == m_table->size() );
    
    for(size_t i=0 ; i < m_table->size() ; i++)
    {
        G4LogicalBorderSurface* lbs = (*m_table)[i] ; 

        LOG(info) << " lbs " << lbs->GetName() ; 

        GBorderSurface* bs = X4LogicalBorderSurface::Convert( lbs );

        assert( bs ); 

        m_slib->add(bs) ;
    }
}


