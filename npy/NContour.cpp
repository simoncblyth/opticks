#include "SPath.hh"
#include "PLOG.hh"
#include "NP.hh"

#include "NBBox.hpp"
#include "NContour.hpp"


void NContour::XZ_bbox_grid( std::vector<float>& xx, std::vector<float>& yy, const nbbox& bb, float sx, float sy, int mx, int my )  // static
{
    float x0 = bb.min.x ; 
    float x1 = bb.max.x ;
    float dx = x1 - x0 ; 
    x0 -= float(mx)*sx*dx ; 
    x1 += float(mx)*sx*dx ; 
 
    float z0 = bb.min.z ; 
    float z1 = bb.max.z ;
    float dz = z1 - z0 ; 
    z0 -= float(my)*sy*dz ; 
    z1 += float(my)*sy*dz ; 

    for(float x=x0 ; x <= x1 ; x+=dx*sx ) xx.push_back(x) ; 
    for(float y=z0 ; y <= z1 ; y+=dz*sy ) yy.push_back(y) ; 
}


NContour::NContour( const std::vector<float>& xx, const std::vector<float>& yy )
    :
    ni(xx.size()),
    nj(yy.size()),
    X(NP::Make<float>(ni, nj)),
    Y(NP::Make<float>(ni, nj)),
    Z(NP::Make<float>(ni, nj)),
    zdat(Z->values<float>()) 
{

    float* xdat = X->values<float>(); 
    float* ydat = Y->values<float>(); 

    for(unsigned i=0 ; i < ni ; i++ )
    {
        float x = xx[i] ; 
        for(unsigned j=0 ; j < nj ; j++ )
        {
            float y = yy[j] ; 
                
            xdat[ nj*i + j ] = x ; 
            ydat[ nj*i + j ] = y ; 
            // prep what np.meshgrid would for matplotlib contour plotting 
        }
    }
}

void NContour::setZ( unsigned i, unsigned j, float z )
{
    assert( i < ni ); 
    assert( j < nj ); 
    zdat[ nj*i + j ] = z ; 
}

void NContour::save(const char* base, const char* rela, const char* relb) const
{
    int create_dirs = 2 ; // 2:dirpath
    const char* fold = SPath::Resolve(base, rela, relb, create_dirs) ; 
    LOG(info) << " fold " << fold ; 

    X->save(fold, "X.npy"); 
    Y->save(fold, "Y.npy"); 
    Z->save(fold, "Z.npy");
} 



