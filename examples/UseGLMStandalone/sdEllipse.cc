/**
sdEllipse.cc : Newton-Raphson numerical solution for closest distance from 2d point to an ellipse
======================================================================================================

The distance to ellipse is computed in a grid and saved as .npy arrays 
See sdEllipse.py for contour plotting. 


* https://www.iquilezles.org/www/articles/ellipsoids/ellipsoids.htm
* https://www.iquilezles.org/www/articles/ellipsedist/ellipsedist.htm

An alternative is to consider a numeric solver. So, let's start computing the
closes point to p on the ellipse, q. We know that the vector p-q will have to
be perpendicular to the tangent of the ellipse at q. So, if we parametrize the
ellipse again as q(ω) = { a⋅cos ω, b⋅sin ω }, then

<p-q(ω), q'(ω)> = 0

with q'(ω) being the tangent, which takes this form:

q'(ω) = { -a⋅sin ω, b⋅cos ω}

If we expand, we get the equation

f(ω) = -a⋅sin ω ⋅(x-a⋅cos ω) + b⋅cos ω ⋅(y-b⋅sin ω) = 0

which we are going to solve by using Newton-Raphson method, for which we need the derivative of the function above:

f'(ω) = -<p-u,v> - <v,v>

with

u = { a⋅cos ω, b⋅sin ω }
v = {-a⋅sin ω, b⋅cos ω }

So our iterative process will be

ωn+1 = ωn + <p-u,v> / (<p-u,u> + <v,v>) 

The most difficult part is ensuring that the initial value of ω, ω0 makes the
sequence converge to the desired root. An reasonable value for ω0 can be
computed by stretching the space so the ellipse becomes a circle, computing the
closest point from the stretched p to the circle, and then undoing the stretch:

ω0 = atan(a⋅y / b⋅x)

Because we are going to work only with the first quadrant of the plane (since
the ellipse is symmetric anyways), this will be fine. In fact, the sequence
will converge for all points in the exterior of the ellipse. Unfortunately,
some point in the interior will not converge, and a different initial ω0 is
required for them. In my implementation, I decided to select all the point
bellow the line passing through the discontinuity of the gradient with slope
(a,b), and make them start at ω0 = 0.


======


    (x/a)^2 + (y/b)^2 = 1    (on the ellipse)

    [x/a, y/b].[x/a, y/b] = 1 


https://www.iquilezles.org/www/articles/ellipsedist/ellipsedist.htm

    float sdEllipse( in vec2 p, in vec2 ab )
    {
        // symmetry
        p = abs( p );
        
        // determine in/out and initial omega value
        bool s = dot(p/ab,p/ab)>1.0;
        float w = s ? atan(p.y*ab.x, p.x*ab.y) : 
                      ((ab.x*(p.x-ab.x)<ab.y*(p.y-ab.y))? 1.5707963 : 0.0);
        
        // find root with Newton solver
        for( int i=0; i<4; i++ )
        {
            vec2 cs = vec2(cos(w),sin(w));
            vec2 u = ab*vec2( cs.x,cs.y);
            vec2 v = ab*vec2(-cs.y,cs.x);
            w = w + dot(p-u,v)/(dot(p-u,u)+dot(v,v));
        }
        
        // compute final point and distance
        return length(p-ab*vec2(cos(w),sin(w))) * (s?1.0:-1.0);
    }



   q(w) = [a cos(w), b sin(w)]

    [x, y ] = [ a cos(w), b sin(w) ]     draw line from origin to the [x,y]

      arctan( a.y/ b.x ) = w

**/


#include <cassert>
#include <sstream>
#include <string>
#include <iostream>
#include <iomanip>
#include <glm/glm.hpp>
#include "NP.hh"

float sdEllipse( const glm::vec2& p_, const glm::vec2& ab, bool debug=false )
{
    glm::vec2 p = glm::abs(p_); 

    float e = glm::dot( p/ab, p/ab );    //  (x/a)^2 + (y/b)^2 = 1  on the ellipse   > 1 outside, < 1 inside
    bool s = e > 1.f ;                   
    float w = s ? 
                    std::atan2f(p.y*ab.x, p.x*ab.y) 
                : 
                    (  (ab.x*(p.x-ab.x)<ab.y*(p.y-ab.y))? 1.5707963 : 0.0)
                ;

    for( int i=0; i<4; i++ )
    {
        glm::vec2 cs = glm::vec2(std::cosf(w),std::sinf(w));  // cos_sin
        glm::vec2 u = ab*glm::vec2( cs.x,cs.y);
        glm::vec2 v = ab*glm::vec2(-cs.y,cs.x);

        if(debug)
        {
            float uv = glm::dot(u, v ); 
            std::cout << " uv " << uv << std::endl ;
        }

        w = w + glm::dot(p-u,v)/(glm::dot(p-u,u)+glm::dot(v,v));
    }
    return glm::length(p-ab*glm::vec2(std::cosf(w),std::sinf(w))) * (s?1.0:-1.0);
}

struct Scan
{
    glm::vec2 ab ;  

    void add(float x, float y, bool dump); 
    void grid(int hx, int hy, float scale);
    void points();

    void save(const char* dir, const char* pfx); 
    std::vector<glm::vec4> recs ; 
};


void Scan::add(float x, float y, bool dump)
{
    glm::vec2 p(x, y); 
    float e = sdEllipse(p, ab, dump); 
    float d = glm::dot( p/ab, p/ab ); 
    recs.push_back({x,y,e,d}) ; 
    if(dump) std::cout 
        << " p( " 
        << std::fixed << std::setw(10) << std::setprecision(4) << p.x 
        << "," 
        << std::fixed << std::setw(10) << std::setprecision(4) << p.y
        << ")"
        << " d:" 
        << std::fixed << std::setw(10) << std::setprecision(4) << d
        << " e:" 
        << std::fixed << std::setw(10) << std::setprecision(4) << e
        << std::endl 
        ;  
}

void Scan::grid(int hx, int hy, float scale)
{
    for(int iy=-hy ; iy <= hy ; iy++) 
    {
        float y = scale*float(iy)/float(hy) ;
        for(int ix=-hx ; ix <= hx ; ix++)
        {
            float x = scale*float(ix)/float(hx) ; 
            add(x,y, false); 
        }
    }
}

void Scan::points()
{
    std::vector<glm::vec2> pp ; 
    pp.push_back( { ab.x,   0.f } );
    pp.push_back( {  0.f, ab.y } );
    pp.push_back( {  0.f, ab.x } );
    pp.push_back( { ab.y,   0.f } );

    for(int i=0 ; i < int(pp.size()) ; i++)
    { 
        const glm::vec2& p = pp[i] ;  
        add(p.x, p.y, true); 
    }
}

void Scan::save(const char* dir, const char* pfx)
{
    std::stringstream ss ; 
    ss << pfx << "_" << ab.x << "_" << ab.y << ".npy" ;
    std::string name = ss.str(); 
    NP::Write(dir, name.c_str(), (float*)recs.data(), recs.size(), 4 ); 
    recs.clear(); 
}

void ellipse_contour_scan( float a, float b  )
{
    Scan scan = {} ; 
    scan.ab = {a, b } ; 

    scan.points(); 
    scan.save("/tmp", "sdEllipse_points"); 

    scan.grid(100, 100, 300.f ); 
    scan.save("/tmp", "sdEllipse_grid"); 
}

int main(int argc, char** argv)
{
    ellipse_contour_scan(254.f, 190.f);  // body ellipsoid is   254_190
    ellipse_contour_scan(249.f, 185.f);  // inner1 ellipsoid is 249_185

    return 0 ; 
}



/**

     (x/a)^2 + (y/b)^2 = 1 

     At x = 0,  y = +-b
     At y = 0,  x = +-a 


In [1]: run gpmt.py                                                                                                                                                                                      
[2021-04-13 12:41:18,425] p96979 {/Users/blyth/opticks/analytic/GDML.py:1239} INFO - parsing gdmlpath $OPTICKS_PREFIX/tds_ngt_pcnk_sycg.gdml 
...

In [6]: lvs[4]                                                                                                                                                                                           
Out[6]: 
[31] Volume HamamatsuR12860_PMT_20inch_inner1_log0x3547fb0
solid
0 [111] Ellipsoid HamamatsuR12860_PMT_20inch_inner1_solid_I0x3560bc0   : xyz 0.0,0.0,0.000   :  ax/by/cz 249.000/249.000/185.000  zcut1  0.000 zcut2 185.000  
material
[11] Material Vacuum0x3365190 gas
physvol 0


In [5]: lvs[3]                                                                                                                                                                                           
Out[5]: 
[33] Volume HamamatsuR12860_PMT_20inch_body_log0x3547d90
solid
0 [139] Union HamamatsuR12860_PMT_20inch_body_solid_1_90x35572e0   : right_xyz:0.0/0.0/-420.000
l:0 [137] Union HamamatsuR12860_PMT_20inch_body_solid_1_80x34c71f0   : right_xyz:0.0/0.0/-385.000
l:0 [135] Union HamamatsuR12860_PMT_20inch_body_solid_1_60x3556000   : right_xyz:0.0/0.0/-275.000
l:0 [133] Union HamamatsuR12860_PMT_20inch_body_solid_1_50x3555cb0   : right_xyz:0.0/0.0/-242.500
l:0 [131] Union HamamatsuR12860_PMT_20inch_body_solid_1_40x354d430   : right_xyz:0.0/0.0/-179.216
l:0 [129] Union HamamatsuR12860_PMT_20inch_body_solid_1_30x354cb30   : right_xyz:0.0/0.0/-5.000
l:0 [127] Union HamamatsuR12860_PMT_20inch_body_solid_1_20x354c7e0   : right_xyz:0.0/0.0/-2.500
l:0 [125] Ellipsoid HamamatsuR12860_PMT_20inch_body_solid_I0x354c5d0   : xyz 0.0,0.0,0.000   :  ax/by/cz 254.000/254.000/190.000  zcut1  0.000 zcut2 190.000  
r:0 [126] Tube HamamatsuR12860_PMT_20inch_body_solid_II0x354c6b0   : xyz 0.0,0.0,5.000   :  rmin 0.0 rmax 254.000 hz  2.500 
r:0 [128] Ellipsoid HamamatsuR12860_PMT_20inch_body_solid_III0x354ca50   : xyz 0.0,0.0,0.000   :  ax/by/cz 254.000/254.000/190.000  zcut1 -190.000 zcut2  0.000  
r:0 [130] Polycone HamamatsuR12860_PMT_20inch_body_solid_IV0x354ce10   : xyz 0.0,0.0,0.000   :  zp_num  2 z:[17.1711421283589, -30.7842712474619] rmax:[142.967108826257, 127.0] rmin:[0.0]  
r:0 [132] Tube HamamatsuR12860_PMT_20inch_body_solid_V0x3555b80   : xyz 0.0,0.0,65.000   :  rmin 0.0 rmax 127.000 hz 32.500 
r:0 [134] Ellipsoid HamamatsuR12860_PMT_20inch_body_solid_VI0x3555f20   : xyz 0.0,0.0,0.000   :  ax/by/cz 127.000/127.000/95.000  zcut1 -90.000 zcut2  0.000  
r:0 [136] Tube HamamatsuR12860_PMT_20inch_body_solid_VIII0x34c70c0   : xyz 0.0,0.0,70.000   :  rmin 0.0 rmax 37.500 hz 35.000 
r:0 [138] Polycone HamamatsuR12860_PMT_20inch_body_solid_IX0x34c74d0   : xyz 0.0,0.0,0.000   :  zp_num  2 z:[0.0, -30.0] rmax:[25.75] rmin:[0.0]  
material
[12] Material Pyrex0x3377e00 solid

**/


