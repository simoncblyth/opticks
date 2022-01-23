#include "SPath.hh"
#include "NP.hh"
#include "scuda.h"
#include "CSGGrid.h"
#include "PLOG.hh"


CSGGrid::CSGGrid( const float4& ce_, int nx_, int ny_, int nz_ )
    :
    ce(ce_),
    margin(1.2f),
    nx(nx_),
    ny(ny_),
    nz(nz_),
    gridscale(make_float3(margin*ce.w/float(nx), margin*ce.w/float(ny), margin*ce.w/float(nz))),
    ni(2*nx+1),
    nj(2*ny+1),
    nk(2*nz+1),
    sdf(NP::Make<float>(ni,nj,nk)),
    sdf_v(sdf->values<float>()),
    xyzd(NP::Make<float>(ni,nj,nk,4)),
    xyzd_v(xyzd->values<float>())
{
    init(); 
}
void CSGGrid::init()
{
    init_meta();
}

void CSGGrid::init_meta()
{
    sdf->set_meta<float>("cex", ce.x  );  
    sdf->set_meta<float>("cey", ce.y  );  
    sdf->set_meta<float>("cez", ce.z  );  
    sdf->set_meta<float>("cew", ce.w  );  

    sdf->set_meta<float>("ox", float(-nx)*gridscale.x );  
    sdf->set_meta<float>("oy", float(-ny)*gridscale.y );  
    sdf->set_meta<float>("oz", float(-nz)*gridscale.z );  

    sdf->set_meta<float>("sx", gridscale.x );  
    sdf->set_meta<float>("sy", gridscale.y );  
    sdf->set_meta<float>("sz", gridscale.z );  
}

void CSGGrid::scan( std::function<float(const float3&)> sdf  )
{
    float3 position = make_float3( 0.f, 0.f, 0.f ); 

    // ZYX ordering used to match pyvista  pv.UniformGrid

    for(int i=0 ; i < ni ; i++ )
    {
        int iz = -nz + i ; 
        position.z = ce.z + float(iz)*gridscale.z ; 

        for(int j=0 ; j < nj ; j++ )
        {
            int iy = -ny + j ;
            position.y = ce.y + float(iy)*gridscale.y ; 

            for(int k=0 ; k < nk ; k++ )
            {
                int ix = -nx + k ;
                position.x = ce.x + float(ix)*gridscale.x ; 
 
                float sd = sdf( position ); 

                int idx = i*nj*nk + j*nk + k ;
  
                xyzd_v[idx*4 + 0] = position.x ; 
                xyzd_v[idx*4 + 1] = position.y ; 
                xyzd_v[idx*4 + 2] = position.z ; 
                xyzd_v[idx*4 + 3] = sd ; 

                sdf_v[idx] = sd ;  
            } 
        }
    }
}
    
const char* CSGGrid::BASE = "$TMP/CSG/CSGSignedDistanceFieldTest" ; 

void CSGGrid::save(const char* geom, const char* base) const 
{    
    int create_dirs = 2 ; // 2:dirpath 
    const char* fold = SPath::Resolve(base ? base : BASE, geom, create_dirs ); 
    LOG(info) << "[ saving sdf.npy " << sdf->sstr() << " to " << fold ; 
    sdf->save(fold, "sdf.npy"); 
    LOG(info) << "]" ; 

    LOG(info) << "[ saving xyzd.npy " << xyzd->sstr() << " to " << fold ; 
    xyzd->save(fold, "xyzd.npy"); 
    LOG(info) << "]" ; 
}



