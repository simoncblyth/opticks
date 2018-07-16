
#include "G4LogicalSkinSurface.hh"

#include "X4LogicalSkinSurfaceTable.hh"
#include "X4LogicalSkinSurface.hh"

#include "GSkinSurface.hh"
#include "GSurfaceLib.hh"

#include "PLOG.hh"


void X4LogicalSkinSurfaceTable::Convert( GSurfaceLib* dst )
{
    X4LogicalSkinSurfaceTable x(dst); 
}

X4LogicalSkinSurfaceTable::X4LogicalSkinSurfaceTable(GSurfaceLib* dst )
    :
    m_src(G4LogicalSkinSurface::GetSurfaceTable()),
    m_dst(dst)
{
    init();
}


void X4LogicalSkinSurfaceTable::init()
{
    unsigned num_src = G4LogicalSkinSurface::GetNumberOfSkinSurfaces() ; 
    assert( num_src == m_src->size() );

    LOG(error) << " NumberOfSkinSurfaces num_src " << num_src ;  
    
    for(size_t i=0 ; i < m_src->size() ; i++)
    {
        G4LogicalSkinSurface* src = (*m_src)[i] ; 

        LOG(info) << " src " << src->GetName() ; 

        GSkinSurface* dst = X4LogicalSkinSurface::Convert( src );

        assert( dst ); 

        m_dst->add(dst) ;
    }
}

