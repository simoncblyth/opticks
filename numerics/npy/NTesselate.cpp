#include "NTesselate.hpp"
#include "NPY.hpp"
#include "NLog.hpp"

#include "GLMPrint.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>


void NTesselate::init()
{ 
    unsigned int ni = m_basis->getShape(0);
    unsigned int nj = m_basis->getShape(1);
    unsigned int nk = m_basis->getShape(2);
    assert(ni > 0 && nj == 3 && nk == 3);
}

struct triangle 
{
    triangle(glm::vec3& a, glm::vec3& b, glm::vec3& c) 
    {
        p[0] = a ; 
        p[1] = b ; 
        p[2] = c ; 
    }

    triangle(float* ptr)
    {
        p[0] = glm::make_vec3(ptr) ; 
        p[1] = glm::make_vec3(ptr+3) ; 
        p[2] = glm::make_vec3(ptr+6) ; 
    }    

    float* values()
    {
         float* ptr = new float[9] ;
         memcpy( ptr+0, glm::value_ptr(p[0]), sizeof(float)*3 );
         memcpy( ptr+3, glm::value_ptr(p[1]), sizeof(float)*3 );
         memcpy( ptr+6, glm::value_ptr(p[2]), sizeof(float)*3 );
         return ptr ; 
    }

    void dump(const char* msg)
    {
        LOG(info) << msg ; 
        print(p[0], "p[0]");
        print(p[1], "p[1]");
        print(p[2], "p[2]");
    }

    glm::vec3 p[3];
}; 



void NTesselate::subdivide(unsigned int nsubdiv)
{
    unsigned int nb = m_basis->getShape(0); // base triangles
    float* basis = m_basis->getValues();

    int ntri = nb*(1 << (nsubdiv * 2));

    LOG(debug) << "NTesselate::subdivide base triangles " << nb << " ntri " << ntri  ; 

    m_tris = NPY<float>::make( 0, 3, 3 ) ;

    for(int s = 0; s < nb ; s++) 
    {
        triangle t(basis + s*3*3);
        subdivide( nsubdiv, t );
    }
    assert(m_tris->getNumItems() == ntri);
}


void NTesselate::subdivide(unsigned int nsubdiv, triangle& t)
{
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
    glm::vec3 x(v) ; 

    a = glm::normalize(a);
    c = glm::normalize(c);
    x = glm::normalize(x); 
   
    triangle t(a,c,x);

    float* vals = t.values();
    m_tris->add(vals, 3*3);

    delete [] vals ; 
}




