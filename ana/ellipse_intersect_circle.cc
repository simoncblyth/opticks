//  gcc ellipse_intersect_circle.cc -lm -lstdc++ -o /tmp/ellipse_intersect_circle  && /tmp/ellipse_intersect_circle
//  head -1 ellipse_intersect_circle.cc | perl -pe 's,//,,' - | sh

#include "ellipse_intersect_circle.hh"

int main(int argc, char** argv)
{
     double e_cx = 0. ; 
     double e_cy = -5. ; 
     double e_ax = 254. ;  
     double e_ay = 190. ;  

     double c_cx = 207. ;   // torus_x 
     double c_cy = -210. ;   // torus_z 
     double c_r = 80. ;  // torus_r

     int n = 1000000 ; 
     bool verbose = false ; 

     Ellipse_Intersect_Circle ec = Ellipse_Intersect_Circle::make( e_cx, e_cy, e_ax, e_ay, c_cx, c_cy, c_r, n, verbose );  

     if(ec.intersect.count > 0)
     printf(" (%10.4f, %10.4f) \n", ec.intersect.p[0].x, ec.intersect.p[0].y ); 

     return 0 ; 
}
