#include "NSphere.hpp"
#include "NTesselate.hpp"
#include "NPY.hpp"
#include <math.h>
#include <glm/glm.hpp>

#include "icosahedron.hpp"

NPY<float>* NSphere::icosahedron(unsigned int nsubdiv)
{
    unsigned int ntris = icosahedron_ntris(nsubdiv);
    float* tris = icosahedron_tris(nsubdiv);

    NPY<float>* buf = NPY<float>::make( ntris, 3, 3); 
    buf->setData(tris);

    //buf->save("/tmp/icosahedron.npy");

    return buf ; 
}


NPY<float>* NSphere::octahedron(unsigned int nsubdiv)
{
    float* data = octahedron_();

    NPY<float>* oct = NPY<float>::make(8, 3, 3);
    oct->setData(data);
    oct->save("/tmp/oct.npy");

    NTesselate* tess = new NTesselate(oct);
    tess->subdivide(nsubdiv);

    NPY<float>* tris = tess->getTriangles();
    return tris ; 
}

template<typename T>
void sincos_(const T angle, T& s, T& c)
{
#ifdef __APPLE__
    __sincos( angle, &s, &c);
#else
    sincos( angle, &s, &c);
#endif
}


NPY<float>* NSphere::latlon(unsigned int n_polar, unsigned int n_azimuthal) 
{
   // TODO: pole handling, make triangles from the quad 

    for(unsigned int i=0 ; i < n_azimuthal ; i++)
    {
        float p0 = 2.0f*M_PI*i/n_azimuthal ;
        float p1 = 2.0f*M_PI*(i+1)/n_azimuthal ;

        double sp0,sp1,cp0,cp1 ;
        sincos_<double>(p0, sp0, cp0 ); 
        sincos_<double>(p1, sp1, cp1 ); 

        for(unsigned int j=0 ; j < n_polar ; j++)
        {
            double t0 = 1.0f*M_PI*i/n_polar ;
            double t1 = 1.0f*M_PI*(i+1)/n_polar ;

            double st0,st1,ct0,ct1 ;
            sincos_<double>(t0, st0, ct0 ); 
            sincos_<double>(t1, st1, ct1 ); 

            // quad on unit sphere
            glm::vec3 p00( st0*cp0, st0*sp0, ct0 );
            glm::vec3 p10( st1*cp0, st1*sp0, ct1 );
            glm::vec3 p01( st0*cp1, st0*sp1, ct0 );
            glm::vec3 p11( st1*cp1, st1*sp1, ct1 );

        }
    } 
    return NULL ; 
}

