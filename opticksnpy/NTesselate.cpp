

#include "NGLM.hpp"
#include "GLMPrint.hpp"

#include "NTesselate.hpp"
#include "NTriangle.hpp"
#include "NTrianglesNPY.hpp"
#include "NPY.hpp"

#include "PLOG.hh"

#ifdef _MSC_VER
// 'ViewNPY' or 'NTrianglesNPY' : object allocated on the heap may not be aligned 16
// https://github.com/g-truc/glm/issues/235
// apparently fixed by 0.9.7.1 Release : currently on 0.9.6.3
#pragma warning( disable : 4316 )
#endif


NTesselate::NTesselate(NPY<float>* basis) 
    :
    m_basis(basis),
    m_tris(NULL)
{
    init();
}



void NTesselate::init()
{ 
    unsigned int ni = m_basis->getShape(0);
    unsigned int nj = m_basis->getShape(1);
    unsigned int nk = m_basis->getShape(2);
    assert(ni > 0 && nj == 3 && nk == 3);
}


NPY<float>* NTesselate::getBuffer()
{
    return m_tris->getTris() ; 
}



void NTesselate::subdivide(unsigned int nsubdiv)
{
    unsigned int nb = m_basis->getShape(0); // base triangles
    float* basis = m_basis->getValues();

    unsigned int ntri = nb*(1 << (nsubdiv * 2));

    LOG(debug) << "NTesselate::subdivide base triangles " << nb << " ntri " << ntri  ; 

    m_tris = new NTrianglesNPY();

    for(unsigned int s = 0; s < nb ; s++) 
    {
        ntriangle t(basis + s*3*3);
        subdivide( nsubdiv, t );
    }
    assert(m_tris->getNumTriangles() == ntri);
}



void NTesselate::subdivide(unsigned int nsubdiv, ntriangle& t)
{
    /*
    Developed from  
    https://www.cosc.brocku.ca/Offerings/3P98/course/OpenGL/glut-3.7/progs/advanced/sphere.c
    by David Blythe, SGI 
    */

    int nrows = 1 << nsubdiv ;
    LOG(debug) << "NTesselate::subdivide nsubdiv " << nsubdiv << " nrows " << nrows  ; 

    for(int i = 0; i < nrows; i++) 
    {
        glm::vec3 v0,v1,v2,v3,va,vb,x1,x2 ;

        v0 = glm::mix( t.p[1], t.p[0], (float)(i+1)/nrows );
        v1 = glm::mix( t.p[1], t.p[0], (float)(i)/nrows );
        v2 = glm::mix( t.p[1], t.p[2], (float)(i+1)/nrows );
        v3 = glm::mix( t.p[1], t.p[2], (float)(i)/nrows );

        x1 = v0 ; 
        x2 = v1 ; 

        for(int j = 0; j < i; j++) 
        {
            va = glm::mix(v0, v2, (float)(j+1)/(i+1) );
            vb = glm::mix(v1, v3, (float)(j+1)/(i) );
             
            add(x1,x2,va);
            x1 = x2; 
            x2 = va;

            add(vb,x2,x1);
            x1 = x2; 
            x2 = vb;
        }
        add(x1, x2, v2);
    }
}



void NTesselate::add(glm::vec3& a, glm::vec3& c, const glm::vec3& v)
{
    // first two points are projected onto unit sphere(aka normalized) in caller context, 
    // third stays constant

    glm::vec3 x(v) ; 

    a = glm::normalize(a);
    c = glm::normalize(c);
    x = glm::normalize(x); 
   
    m_tris->add(a,c,x); 
}



