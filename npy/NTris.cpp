#include <iostream>
#include <sstream>
#include <iomanip>
#include <cassert>

#include <boost/math/constants/constants.hpp>

#include "NGLMExt.hpp"
#include "NTris.hpp"

void NTris::add( const glm::vec3& a, const glm::vec3& b, const glm::vec3& c)
{
    unsigned i = verts.size(); verts.push_back(a); 
    unsigned j = verts.size(); verts.push_back(b); 
    unsigned k = verts.size(); verts.push_back(c); 
    glm::uvec3 tri(i,j,k);
    tris.push_back(tri);
}  

unsigned NTris::get_num_tri() const 
{
    return tris.size();
}

unsigned NTris::get_num_vert() const 
{
    return verts.size();
}


void NTris::get_vert( unsigned i, glm::vec3& v ) const 
{
    v = verts[i] ; 
}
void NTris::get_normal( unsigned /*i*/, glm::vec3& n ) const 
{
    n.x = 0  ; 
    n.y = 0  ; 
    n.z = 0  ; 
}
void NTris::get_uv( unsigned /*i*/, glm::vec3& uv ) const 
{
    uv.x = 0  ; 
    uv.y = 0  ; 
    uv.z = 0  ; 
}



void NTris::get_tri(unsigned i, glm::uvec3& t) const 
{
    t = tris[i] ; 
}
void NTris::get_tri(unsigned i, glm::uvec3& t, glm::vec3& a, glm::vec3& b, glm::vec3& c ) const 
{
    t = tris[i] ; 
    a = verts[t.x]; 
    b = verts[t.y]; 
    c = verts[t.z]; 
}



std::string NTris::brief() const 
{
    std::stringstream ss ; 

    ss << "NTris"
       << " nf " << std::setw(5) << tris.size() 
       << " nv " << std::setw(5) << verts.size() 
       ;

    return ss.str();
}

void NTris::dump(const char* msg) const 
{
    std::cout << msg << " " << brief() << std::endl ;

    glm::uvec3 t ; 
    glm::vec3 a ; 
    glm::vec3 b ; 
    glm::vec3 c ; 

    unsigned ntri = get_num_tri();
    for(unsigned i=0 ; i < ntri ; i++)
    {
        get_tri(i, t, a,b,c );

        std::cout << " t " << std::setw(20) << glm::to_string(t) 
                  << " a " << std::setw(20) << glm::to_string(a) 
                  << " b " << std::setw(20) << glm::to_string(b)
                  << " c " << std::setw(20) << glm::to_string(c)
                  << std::endl
                  ; 

    }

}


NTris* NTris::make_sphere( unsigned n_polar, unsigned n_azimuthal, float ctmin, float ctmax ) 
{

    /*
 
                                  t0             t1            ct0       

            t = 0                  0             1/n_polar      +1   

            t = n_polar - 1      1 - 1/n_polar     1            -1   



                 -  ct0
                    max_exclude:  ct1 > zmax 
                 -  ct1               - ct0
    zmax     +-----------------+        max_straddle
            /    -              \     - ct1
           /                     \
          /      -                \   - ct0
    zmin +-------------------------+    min_straddle
                 - ct0                - ct1
                   min_exclude:  ct0 < zmin
                 - ct1

    */
 
    assert(ctmax > ctmin && ctmax <= 1.f && ctmin >= -1.f);


    NTris* tris = new NTris ; 

    float pi = boost::math::constants::pi<float>() ;

    for(unsigned t=0 ; t < n_polar ; t++)
    {
        double t0 = 1.0f*pi*float(t)/n_polar ; 
        double t1 = 1.0f*pi*float(t+1)/n_polar ;

        double st0,st1,ct0,ct1 ;
        sincos_<double>(t0, st0, ct0 ); 
        sincos_<double>(t1, st1, ct1 ); 
        assert( ct0 > ct1 );

        bool max_exclude  = ct1 > ctmax ;
        bool max_straddle = ctmax <= ct0 && ctmax > ct1 ; 
        bool min_straddle = ctmin <= ct0 && ctmin > ct1 ; 
        bool min_exclude  = ct0 < ctmin ;  

        if(max_exclude || min_exclude ) 
        {
            continue ; 
        }
        else if(max_straddle)
        {
            sincos_<double>(acos(ctmax), st0, ct0 ); 
        }
        else if(min_straddle) 
        {
            sincos_<double>(acos(ctmin), st1, ct1 ); 
        }

        for(unsigned p=0 ; p < n_azimuthal ; p++)
        {
            float p0 = 2.0f*pi*float(p)/n_azimuthal ;
            float p1 = 2.0f*pi*float(p+1)/n_azimuthal ;

            double sp0,sp1,cp0,cp1 ;
            sincos_<double>(p0, sp0, cp0 ); 
            sincos_<double>(p1, sp1, cp1 ); 

            glm::vec3 x00( st0*cp0, st0*sp0, ct0 );
            glm::vec3 x10( st0*cp1, st0*sp1, ct0 );

            glm::vec3 x01( st1*cp0, st1*sp0, ct1 );
            glm::vec3 x11( st1*cp1, st1*sp1, ct1 );

   
            if( t == 0 || t == n_polar - 1) 
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




