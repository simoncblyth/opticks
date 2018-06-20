
#include <boost/math/constants/constants.hpp>

#include "NGLMExt.hpp"
#include "GLMPrint.hpp"
#include "NPY.hpp"

#include "NQuad.hpp"
#include "NBBox.hpp"
#include "NTrianglesNPY.hpp"
#include "NTesselate.hpp"
#include "NTriangle.hpp"

#include "NTris.hpp"

#include "PLOG.hh"

#ifdef _MSC_VER
// instanciations of 'NTrianglesNPY' raise warning:
//    object allocated on the heap may not be aligned 16
// https://github.com/g-truc/glm/issues/235
// apparently fixed by 0.9.7.1 Release : currently on 0.9.6.3

#pragma warning( disable : 4316 )
#endif


const char* NTrianglesNPY::PLACEHOLDER = "PLACEHOLDER" ; 



NPY<float>* NTrianglesNPY::getTris() const 
{
    return m_tris ; 
}
NPY<float>* NTrianglesNPY::getNormals() const 
{
    return m_normals ; 
}

float NTrianglesNPY::maxdiff( const NTrianglesNPY* other, bool dump )
{
    return m_tris->maxdiff( other->getTris(), dump ); 
}



void NTrianglesNPY::dump(const char* msg) const 
{
    LOG(info) << "NTrianglesNPY::dump"
              << " tris " << m_tris->getShapeString()
              ;

    m_tris->dump(msg);
}


void NTrianglesNPY::setPoly(const std::string& poly)
{
   m_poly = poly ; 
}
const std::string& NTrianglesNPY::getPoly()
{
   return m_poly ; 
}



void NTrianglesNPY::setMessage(const std::string& msg)
{
   m_message = msg ; 
}
const std::string& NTrianglesNPY::getMessage()
{
   return m_message ; 
}

bool NTrianglesNPY::hasMessage(const std::string& msg)
{
    return m_message.compare(msg) == 0 ; 
}
bool NTrianglesNPY::isPlaceholder()
{
    return m_message.compare(PLACEHOLDER) == 0 ; 
}







void NTrianglesNPY::setTransform(const glm::mat4& transform_)
{
    m_transform = transform_ ; 
}
glm::mat4 NTrianglesNPY::getTransform()
{
    return m_transform ; 
}



NTrianglesNPY::NTrianglesNPY(const NTriSource* tris)
    :
    m_tris(NPY<float>::make(0,3,3)),
    m_normals(NPY<float>::make(0,3,3))
{
    glm::uvec3 t ; 
    glm::vec3 a ; 
    glm::vec3 b ; 
    glm::vec3 c ; 
    
    unsigned ntri = tris->get_num_tri();
    for(unsigned i=0 ; i < ntri ; i++)
    {   
        tris->get_tri(i,t,a,b,c );
        add(a,b,c); 
    }
}


NTrianglesNPY::NTrianglesNPY() 
   :
    m_tris(NPY<float>::make(0,3,3)),
    m_normals(NPY<float>::make(0,3,3))
{
}

NTrianglesNPY::NTrianglesNPY(NPY<float>* tris, NPY<float>* normals)
   :
   m_tris(tris),
   m_normals(normals)
{
}

NTrianglesNPY* NTrianglesNPY::transform(glm::mat4& m)
{
    NPY<float>* tbuf = m_tris->transform(m);

    NTrianglesNPY* t = new NTrianglesNPY(tbuf);

    return t ; 
}

NTrianglesNPY* NTrianglesNPY::subdivide(unsigned int nsubdiv)
{
    NTesselate* tess = new NTesselate(m_tris);

    tess->subdivide(nsubdiv);

    NPY<float>* tbuf = tess->getBuffer();

    NTrianglesNPY* t = new NTrianglesNPY(tbuf);

    t->setTransform( getTransform() );

    return t ;
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


void NTrianglesNPY::addNormal(const glm::vec3& nrm )  // in triplicate
{
    int n = 3*3 ; 
    float* vals = new float[n] ;
    for(int i=0 ; i < 3 ; i++)  
    {
       vals[i*3+0] = nrm.x ;
       vals[i*3+1] = nrm.y ;
       vals[i*3+2] = nrm.z ;
    }
    m_normals->add(vals, n);
    delete [] vals ; 
}


void NTrianglesNPY::addNormal(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c ) 
{
    float vals[9] ;

    vals[0*3 + 0 ] = a.x ; 
    vals[0*3 + 1 ] = a.y ; 
    vals[0*3 + 2 ] = a.z ;

    vals[1*3 + 0 ] = b.x ; 
    vals[1*3 + 1 ] = b.y ; 
    vals[1*3 + 2 ] = b.z ;
  
    vals[2*3 + 0 ] = c.x ; 
    vals[2*3 + 1 ] = c.y ; 
    vals[2*3 + 2 ] = c.z ;
 
    m_normals->add(vals, 9);
}




void NTrianglesNPY::add(const ntriangle& t )
{
    unsigned int n = 3*3 ; 
    float* vals = new float[n] ;
    t.copyTo(vals); 
    m_tris->add(vals, n);
    delete [] vals ; 
}


void NTrianglesNPY::add(NTrianglesNPY* other )
{
    m_tris->add(other->getTris());
}

nbbox* NTrianglesNPY::findBBox()
{
    NPY<float>* buf = getTris();
    assert(buf && buf->hasItemShape(3,3));
    unsigned nitem = buf->getShape(0);
    nbbox* bb = NULL ; 
    if(nitem > 0)
    {
        ntrange3<float> r = buf->minmax3(); 
        bb = new nbbox ;  
        bb->min.x = r.min.x ; 
        bb->min.y = r.min.y ; 
        bb->min.z = r.min.z ;
     
        bb->max.x = r.max.x ; 
        bb->max.y = r.max.y ; 
        bb->max.z = r.max.z ;
    }
    return bb ; 
}


/* for icosahedron 

* 20 faces (triangular)
* 12 verts
* 30 edges

*/
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




NTrianglesNPY* NTrianglesNPY::from_indexed( NPY<float>* vtx, NPY<unsigned>* idx )
{
    assert( vtx->hasShape( -1, 3) && idx->hasShape(-1,3) );
    // each idx entry of 3 unsigned ints points to three vertices of the triangle
     
    unsigned nvtx = vtx->getShape(0) ; 
    unsigned ntri = idx->getShape(0) ; 

    LOG(info) << " idx " << idx->getShapeString() ; 
    LOG(info) << " vtx " << vtx->getShapeString() ; 

    NTrianglesNPY* tris = new NTrianglesNPY();

    for(unsigned i=0 ; i < ntri ; i++)
    {
        unsigned v0 = idx->getValue( i, 0, 0,  0 );
        unsigned v1 = idx->getValue( i, 0, 0,  1 );
        unsigned v2 = idx->getValue( i, 0, 0,  2 );
 
        assert( v0 < nvtx );
        assert( v1 < nvtx );
        assert( v2 < nvtx );

        glm::vec3 vtx0 ; 
        vtx0.x = vtx->getValue( v0, 0, 0,  0 );
        vtx0.y = vtx->getValue( v0, 0, 0,  1 );
        vtx0.z = vtx->getValue( v0, 0, 0,  2 );

        glm::vec3 vtx1 ; 
        vtx1.x = vtx->getValue( v1, 0, 0,  0 );
        vtx1.y = vtx->getValue( v1, 0, 0,  1 );
        vtx1.z = vtx->getValue( v1, 0, 0,  2 );

        glm::vec3 vtx2 ; 
        vtx2.x = vtx->getValue( v2, 0, 0,  0 );
        vtx2.y = vtx->getValue( v2, 0, 0,  1 );
        vtx2.z = vtx->getValue( v2, 0, 0,  2 );

        tris->add( vtx0, vtx1, vtx2 );
    }
    return tris ;
}

void NTrianglesNPY::to_vtxidx(NVtxIdx& vtxidx) 
{
    assert( m_tris->hasShape(-1,3,3) );  // items with 3*3 float for each tri
    unsigned ntri = m_tris->getShape(0) ; 

    NPY<float>* vtx = NPY<float>::copy( m_tris ) ;
    vtx->reshape( ntri*3 , 3 );

    NPY<unsigned>* idx = NPY<unsigned>::make( ntri, 3 ) ;
    idx->zero();

    vtxidx.vtx = vtx ;   // duplicate vertices for each use in a face     
    vtxidx.idx = idx ; 


    for(unsigned i=0 ; i < ntri ; i++)
    {
        idx->setValue( i, 0, 0,  0,  i*3+0  );
        idx->setValue( i, 0, 0,  1,  i*3+1  );
        idx->setValue( i, 0, 0,  2,  i*3+2  );
    }
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

NTrianglesNPY* NTrianglesNPY::hemi_octahedron()
{
    NTrianglesNPY* tris = new NTrianglesNPY();

    // PZ pyramid
    tris->add(PX,PZ,MY);
    tris->add(MY,PZ,MX);
    tris->add(MX,PZ,PY);
    tris->add(PY,PZ,PX);

    return tris ;
}





NTrianglesNPY* NTrianglesNPY::sphere(unsigned int n_polar, unsigned int n_azimuthal) 
{
    glm::vec4 param(-1.f, 1.f, 0.f, 1.f);
    return sphere(param, n_polar, n_azimuthal);
}


void NTrianglesNPY::setTransform(const glm::vec3& scale, const glm::vec3& translate)
{
    glm::mat4 m_scale = glm::scale(glm::mat4(1.0f), scale);
    glm::mat4 m_translate = glm::translate(glm::mat4(1.0f), translate);
    glm::mat4 mat = m_translate * m_scale ; 

    //print(mat, "NTrianglesNPY::setTransform");

    setTransform(mat);
}



NTrianglesNPY* NTrianglesNPY::sphere(glm::vec4& param, unsigned int n_polar, unsigned int n_azimuthal) 
{
    float ctmin = param.x ; 
    float ctmax = param.y ; 
    float zpos  = param.z ; 
    float radius = param.w ; 

    // unit sphere at origin 
    NTris* ts = NTris::make_sphere( n_polar, n_azimuthal, ctmin, ctmax );

    NTrianglesNPY* tris = new NTrianglesNPY(ts); 

    glm::vec3 scale(radius);
    glm::vec3 translate(0,0,zpos);
    tris->setTransform(scale, translate);   

    // dont apply the transform yet, as whilst sphere at origin
    // can easily get normals from the positions

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





NTrianglesNPY* NTrianglesNPY::disk(glm::vec4& param, unsigned int n_azimuthal) 
{
    float ct = param.x ; 
    double ct0, st0 ;
    sincos_<double>(acos(ct), st0, ct0 ); 
    float pi = boost::math::constants::pi<float>() ;

    NTrianglesNPY* tris = new NTrianglesNPY();
    for(unsigned int p=0 ; p < n_azimuthal ; p++)
    {
        float p0 = 2.0f*pi*float(p)/n_azimuthal ;
        float p1 = 2.0f*pi*float(p+1)/n_azimuthal ;

        double sp0,sp1,cp0,cp1 ;
        sincos_<double>(p0, sp0, cp0 ); 
        sincos_<double>(p1, sp1, cp1 ); 

        glm::vec3 x0(   st0*cp0,  st0*sp0,   ct0 );
        glm::vec3 x1(   st0*cp1,  st0*sp1,   ct0 );
        glm::vec3 xc(         0,        0,   ct0 );

        tris->add(x0,x1,xc); // winding order directs normal along -z 
    }
    return tris ; 
}







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




NTrianglesNPY* NTrianglesNPY::box(const nbbox& bb)
{
    NTrianglesNPY* tris = new NTrianglesNPY();
   
    glm::vec3 P(bb.max.x, bb.max.y, bb.max.z );
    glm::vec3 M(bb.min.x, bb.min.y, bb.min.z );

    glm::vec3 _PXPYPZ( P.x, P.y, P.z ); 
    glm::vec3 _PXPYMZ( P.x, P.y, M.z ); 
    glm::vec3 _PXMYPZ( P.x, M.y, P.z );
    glm::vec3 _PXMYMZ( P.x, M.y, M.z );

    glm::vec3 _MXPYPZ( M.x, P.y, P.z );
    glm::vec3 _MXPYMZ( M.x, P.y, M.z );
    glm::vec3 _MXMYPZ( M.x, M.y, P.z );
    glm::vec3 _MXMYMZ( M.x, M.y, M.z ) ;

    tris->add(_PXPYPZ, _MXPYPZ, _MXMYPZ, _PXMYPZ);    // PZ face
    tris->add(_PXPYPZ, _PXMYPZ, _PXMYMZ, _PXPYMZ);    // PX face
    tris->add(_PXPYPZ, _PXPYMZ, _MXPYMZ, _MXPYPZ);    // PY face   

    tris->add(_MXMYMZ, _PXMYMZ, _PXPYMZ, _MXPYMZ);    // MZ face  
    tris->add(_MXMYMZ, _MXPYMZ, _MXPYPZ, _MXMYPZ);    // MX face
    tris->add(_MXMYMZ, _MXMYPZ, _PXMYPZ, _PXMYMZ);    // MY face

    return tris ; 
}







NTrianglesNPY* NTrianglesNPY::prism(const glm::vec4& param)
{
/*
::

   .                
               
               A'
                    A  (0,height)
                   /|\
                  / | \
                 /  |  \
                /   h   \
               /    |    \ (x,y)   
         L'   M     |     N   
             /      |      \
            L-------O-------R   
                               
     (-hwidth,0)  (0,0)    (hwidth, 0)     

*/

    // hmm how to avoid duplication between here and hemi-pmt.cu/make_prism
    float angle = param.x > 0.f ? param.x : 60.f ; 
    float height = param.y > 0.f ? param.y : param.w  ;
    float depth = param.z > 0.f ? param.z : param.w ;

    float pi = boost::math::constants::pi<float>() ;
    float hwidth = height*tan((pi/180.f)*angle/2.0f) ;      

    float ymax =  height/2.0f ; 
    float ymin = -height/2.0f ; 

    glm::vec3 apex_near(0.f,     ymax, depth/2.f );
    glm::vec3 apex_far( 0.f,     ymax, -depth/2.f );

    glm::vec3 left_near(-hwidth,  ymin, depth/2.f);    
    glm::vec3 left_far(-hwidth,   ymin, -depth/2.f);    

    glm::vec3 right_near( hwidth, ymin, depth/2.f);    
    glm::vec3 right_far( hwidth,  ymin, -depth/2.f);    

    NTrianglesNPY* tris = new NTrianglesNPY();

    tris->add( apex_near, apex_far, left_far, left_near );
    tris->add( apex_near, right_near,  right_far,  apex_far );
    tris->add( left_near, left_far, right_far, right_near );
    tris->add( apex_near, left_near, right_near );
    tris->add( apex_far,  right_far, left_far );

    return tris ; 
}



