
#include "G4LogicalBorderSurface.hh"

#include "X4LogicalBorderSurfaceTable.hh"
#include "X4LogicalBorderSurface.hh"

#include "GBorderSurface.hh"
#include "GSurfaceLib.hh"

#include "PLOG.hh"


void X4LogicalBorderSurfaceTable::Convert( GSurfaceLib* dst )
{
    X4LogicalBorderSurfaceTable xtab(dst); 
}

X4LogicalBorderSurfaceTable::X4LogicalBorderSurfaceTable(GSurfaceLib* dst )
    :
    m_src(G4LogicalBorderSurface::GetSurfaceTable()),
    m_dst(dst)
{
    init();
}


void X4LogicalBorderSurfaceTable::init()
{
    unsigned num_src = G4LogicalBorderSurface::GetNumberOfBorderSurfaces() ; 
    assert( num_src == m_src->size() );

    LOG(debug) << " NumberOfBorderSurfaces " << num_src ;  
    
    for(size_t i=0 ; i < m_src->size() ; i++)
    {
        G4LogicalBorderSurface* src = (*m_src)[i] ; 

        //LOG(info) << src->GetName() ; 

        GBorderSurface* dst = X4LogicalBorderSurface::Convert( src );

        assert( dst ); 

        m_dst->add(dst) ; // GSurfaceLib
    }
}


