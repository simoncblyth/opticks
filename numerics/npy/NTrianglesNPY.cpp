#include "NTrianglesNPY.hpp"
#include "NTesselate.hpp"
#include "NTriangle.hpp"
#include "NPY.hpp"


NTrianglesNPY::NTrianglesNPY(unsigned int n)
{
    m_tris = NPY<float>::make(n,3,3);
}

unsigned int NTrianglesNPY::getNumTriangles()
{
    return m_tris->getNumItems();
}


void NTrianglesNPY::add(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, const glm::vec3& d)
{
   /*
         a------d
         |    . | 
         |  .   |
         |.     |
         b------c

   */
    add(a,b,d);
    add(d,b,c);
}

void NTrianglesNPY::add(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c)
{
    ntriangle t(a,b,c);
    add(t);
}


void NTrianglesNPY::add(const ntriangle& t )
{
    unsigned int n = 3*3 ; 
    float* vals = new float[n] ;
    t.copyTo(vals); 
    m_tris->add(vals, n);
    delete [] vals ; 
}


NPY<float>* NTrianglesNPY::subdivide(unsigned int nsubdiv)
{
    NTesselate* tess = new NTesselate(m_tris);

    tess->subdivide(nsubdiv);

    NPY<float>* buf = tess->getBuffer();

    return buf ;
}





/* for icosahedron */
#define CZ (0.89442719099991)   /*  2/sqrt(5) */
#define SZ (0.44721359549995)   /*  1/sqrt(5) */
#define C1 (0.951056516)        /* cos(18),  */
#define S1 (0.309016994)        /* sin(18) */
#define C2 (0.587785252)        /* cos(54),  */
#define S2 (0.809016994)        /* sin(54) */
#define X1 (C1*CZ)
#define Y1 (S1*CZ)
#define X2 (C2*CZ)
#define Y2 (S2*CZ)

const glm::vec3 NTrianglesNPY::Ip0 = glm::vec3(0,0,1.) ;
const glm::vec3 NTrianglesNPY::Ip1 = glm::vec3(-X2,-Y2,SZ) ;
const glm::vec3 NTrianglesNPY::Ip2 = glm::vec3( X2,-Y2,SZ) ;
const glm::vec3 NTrianglesNPY::Ip3 = glm::vec3( X1, Y1,SZ) ;
const glm::vec3 NTrianglesNPY::Ip4 = glm::vec3(  0, CZ,SZ) ;
const glm::vec3 NTrianglesNPY::Ip5 = glm::vec3(-X1, Y1,SZ) ;

const glm::vec3 NTrianglesNPY::Im0 = glm::vec3(-X1, -Y1,-SZ) ;
const glm::vec3 NTrianglesNPY::Im1 = glm::vec3(  0, -CZ,-SZ) ;
const glm::vec3 NTrianglesNPY::Im2 = glm::vec3( X1, -Y1,-SZ) ;
const glm::vec3 NTrianglesNPY::Im3 = glm::vec3( X2,  Y2,-SZ) ;
const glm::vec3 NTrianglesNPY::Im4 = glm::vec3(-X2,  Y2,-SZ) ;
const glm::vec3 NTrianglesNPY::Im5 = glm::vec3(0,0,-1.) ;


NTrianglesNPY* NTrianglesNPY::icosahedron()
{
    NTrianglesNPY* tris = new NTrianglesNPY();

    /* front pole */
    tris->add(Ip0, Ip1, Ip2);
    tris->add(Ip0, Ip5, Ip1);
    tris->add(Ip0, Ip4, Ip5);
    tris->add(Ip0, Ip3, Ip4);
    tris->add(Ip0, Ip2, Ip3);

    /* mid */
    tris->add(Ip1, Im0, Im1);
    tris->add(Im0, Ip1, Ip5);
    tris->add(Ip5, Im4, Im0);
    tris->add(Im4, Ip5, Ip4);
    tris->add(Ip4, Im3, Im4);
    tris->add(Im3, Ip4, Ip3);
    tris->add(Ip3, Im2, Im3);
    tris->add(Im2, Ip3, Ip2);
    tris->add(Ip2, Im1, Im2);
    tris->add(Im1, Ip2, Ip1);

    /* back pole */
    tris->add(Im3, Im2, Im5);
    tris->add(Im4, Im3, Im5);
    tris->add(Im0, Im4, Im5);
    tris->add(Im1, Im0, Im5);
    tris->add(Im2, Im1, Im5);

    return tris ;
}




const glm::vec3 NTrianglesNPY::PX = glm::vec3(1,0,0) ;
const glm::vec3 NTrianglesNPY::PY = glm::vec3(0,1,0) ;
const glm::vec3 NTrianglesNPY::PZ = glm::vec3(0,0,1) ;
const glm::vec3 NTrianglesNPY::MX = glm::vec3(-1,0,0) ;
const glm::vec3 NTrianglesNPY::MY = glm::vec3(0,-1,0) ;
const glm::vec3 NTrianglesNPY::MZ = glm::vec3(0,0,-1) ;

NTrianglesNPY* NTrianglesNPY::octahedron()
{
    NTrianglesNPY* tris = new NTrianglesNPY();

    // PZ pyramid
    tris->add(PX,PZ,MY);
    tris->add(MY,PZ,MX);
    tris->add(MX,PZ,PY);
    tris->add(PY,PZ,PX);

    // MZ pyramid
    tris->add(PX,MY,MZ);
    tris->add(MY,MX,MZ);
    tris->add(MX,PY,MZ);
    tris->add(PY,PX,MZ);

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


NTrianglesNPY* NTrianglesNPY::sphere(unsigned int n_polar, unsigned int n_azimuthal) 
{
    NTrianglesNPY* tris = new NTrianglesNPY();

    for(unsigned int t=0 ; t < n_polar ; t++)
    {
                                                   //  t = 0     -->    t = n_polar - 1    
        double t0 = 1.0f*M_PI*float(t)/n_polar ;   //     0      -->  (n_polar-1)/n_polar =  1 - 1/n_polar
        double t1 = 1.0f*M_PI*float(t+1)/n_polar ; // 1/n_polar  -->   n_polar/n_polar    =  1

        double st0,st1,ct0,ct1 ;
        sincos_<double>(t0, st0, ct0 ); 
        sincos_<double>(t1, st1, ct1 ); 

        for(unsigned int p=0 ; p < n_azimuthal ; p++)
        {
            float p0 = 2.0f*M_PI*float(p)/n_azimuthal ;
            float p1 = 2.0f*M_PI*float(p+1)/n_azimuthal ;

            double sp0,sp1,cp0,cp1 ;
            sincos_<double>(p0, sp0, cp0 ); 
            sincos_<double>(p1, sp1, cp1 ); 

            // quad on unit sphere
            glm::vec3 x00( st0*cp0, st0*sp0, ct0 );
            glm::vec3 x10( st0*cp1, st0*sp1, ct0 );

            glm::vec3 x01( st1*cp0, st1*sp0, ct1 );
            glm::vec3 x11( st1*cp1, st1*sp1, ct1 );

            if( t == 0 ) 
            {
                tris->add(x00,x01,x11);
            }
            else if( t == n_polar - 1)
            {
                tris->add(x00,x01,x11);
            }
            else
            {
                tris->add(x00,x01,x10);
                tris->add(x10,x01,x11); 
            }
        }
    } 
    return tris ; 
}

/*
         p0     p1

     t0  x00----x10
          |   .  |
          | .    |
     t1  x01----x11


      t = 0  => t0 = 0 
               x00 x10 degenerate (0,0,1)  
 
      t = n_polar - 1 => t1 = pi   
               x01 x11 degenerate (0,0,-1)  

*/




const glm::vec3 NTrianglesNPY::PXPYPZ = glm::vec3(1,1,1) ;
const glm::vec3 NTrianglesNPY::PXPYMZ = glm::vec3(1,1,-1) ;
const glm::vec3 NTrianglesNPY::PXMYPZ = glm::vec3(1,-1,1) ;
const glm::vec3 NTrianglesNPY::PXMYMZ = glm::vec3(1,-1,-1) ;

const glm::vec3 NTrianglesNPY::MXPYPZ = glm::vec3(-1,1,1) ;
const glm::vec3 NTrianglesNPY::MXPYMZ = glm::vec3(-1,1,-1) ;
const glm::vec3 NTrianglesNPY::MXMYPZ = glm::vec3(-1,-1,1) ;
const glm::vec3 NTrianglesNPY::MXMYMZ = glm::vec3(-1,-1,-1) ;


NTrianglesNPY* NTrianglesNPY::cube()
{
    NTrianglesNPY* tris = new NTrianglesNPY();

    tris->add(PXPYPZ, MXPYPZ, MXMYPZ, PXMYPZ);    // PZ face
    tris->add(PXPYPZ, PXMYPZ, PXMYMZ, PXPYMZ);    // PX face
    tris->add(PXPYPZ, PXPYMZ, MXPYMZ, MXPYPZ);    // PY face   

    tris->add(MXMYMZ, PXMYMZ, PXPYMZ, MXPYMZ);    // MZ face  
    tris->add(MXMYMZ, MXPYMZ, MXPYPZ, MXMYPZ);    // MX face
    tris->add(MXMYMZ, MXMYPZ, PXMYPZ, PXMYMZ);    // MY face

    return tris ; 
}




