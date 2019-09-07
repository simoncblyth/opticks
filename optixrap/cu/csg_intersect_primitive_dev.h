/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

// failed attempts to optimize torus "inplace" 




#ifdef TORUS_DEBUG 
    const float a = qn[3] ; 
    const float b = qn[2] ; 
    const float c = qn[1] ; 
    const float d = qn[0] ; 

    // coeff of depressed quartic
    const float e     = b - 3.f * a * a / 8.f;
    const float f     = c + a * a * a / 8.f - 0.5f * a * b;
    const float g     = d - 3.f * a * a * a * a / 256.f + a * a * b / 16.f - a * c / 4.f;

    // switch x -> -x and multiplying by -1 , will flip even x^2, x^0 coeff signs    

    float neumark[3] ;  // coeff of resolvent cubic
    neumark[2] = 2.f*e ; 
    neumark[1] = e*e - 4.f*g ;
    neumark[0] = -f*f ;

    float neumark_delta2 = cubic_delta2( 1.f, neumark[2], neumark[1], neumark[0] ); 
    bool double_root = fabsf( neumark_delta2 - 1.0f ) < 1.e-3f ; 

#endif



/*
   // results in funny cuts


      Check intersect with cylinder just fitting in hole of torus

           x^2 + y^2 - rmin_rmin = 0     (for all z, actually can constrain z in -r->r ) 

           (ox+sx t)^2 + (oy + sy t)^2 - rmin_rmin = 0 

           ox^2+oy^2 - rmin_rmin + 2t( ox sx + oy sy ) + (sx^2 + sy^2) t^2  = 0 

           disc =    oxsx_oysy*oxsx_oysy - sxsx_sysy*(oxox_oyoy - rmin_rmin) 


      Rays with closest approach within rmin will usually intersect the 
      cylinder : need to pick the ones with |z_intersect| > r  of the intersects... 




    if( rcrc < rmin_rmin )  
    {
        // quadratic coeffs of infinite cylinder  d t^2 + 2t b + c = 0 
        const float d = sxsx_sysy ;
        const float b = oxsx_oysy ;
        const float c = oxox_oyoy - rmin_rmin ;
        const float disc  =  b*b - d*c ; 
        const float sdisc = disc > 0.f ? sqrt(disc) : 0.f ; 
        const float q = b > 0.f ? -(b + sdisc) : -(b - sdisc) ;   // avoid non-robust subtraction
        const float t1 = q/d  ; 
        const float t2 = c/q  ;   
        const float z1 = fabsf(oz+t1*sz) ; 
        const float z2 = fabsf(oz+t2*sz) ; 

        if( disc > 0.f && z1 > r && z2 > r )  
        {
            return false ; 
        }
    }
*/



     /*
 
      * rotional symmetry of torus means can rotate ray into 
        convenient orientation without changing intersects
 
      * rotate ray around z axis (torus axis) to make s.x very small, 
        ie rotate ray into plane parallel to z-y axes
                 
                    X
                    |
                 ________       _________ (R+r) 
                /        \      _________
               /   ____   \     _________  
              /   /    \   \               R-r 
             /   /  *---\---\-----------------------
      Y __  /   /        \   \  _________  0 
            \   \        /   /
             \   \      /   /             
              \   \____/   /    _________ -(R-r)
               \          /     _________ 
                \________/      _________ 
                                          -(R+r)
                    
         Vaclav Skala "Line-Torus Implentation for Ray Tracing : Alternative Formulations"

         Torus is viewed as the sweep of an infinite number of spheres, 
         restricting that to the plane of the ray, the torus profile 
         results from an envelope of many circles from the phi sweep.
  
         Cross sectional view, in plane of rotated ray  x=xc=xoffset
  

                                Z        /
                                |       /
                                       /
               _____                  /       _____
              /     \                /       /     \
             /       \          O   *       /       \      -> Y
             \       /             /        \       /
              \_____/             /          \_____/
                                 /
                                /
                               /    *      |    |
                                    y0     yb   ya=yb+r
                                               

           Ray in the plane
                 [ox, oy, oz] + t*[->0, sy, sz]              

           Mid-envelope circle
                 (y - ya)^2 + z^2 - r^2 = 0 
          
                 (oy - ya + t sy )^2  + (oz + t*sz )^2 - r^2 = 0   

                 (sy^2 + sz^2) t^2 +  2*t*[(oy-ya)*sy + (oz*sz)] + (oy-ya)^2 + oz^2 - r^2 = 0 
                 
           Hmm is it valid to just check for ray intersect with mid envelope circle ?
        */

/*
       // cuts more that should...
        else if( xoffset < inner )  // may go thru hole
        {  
            // y0 : ray intersect with torus plane (z=0) 
            float y0 = s.z == 0.f ? RT_DEFAULT_MAX  : o.y + (-o.z/s.z)*s.y ;   
            float yb = sqrt(inner*inner - xoffset*xoffset) ; 
            float ya = yb + r ; 

            if(fabs(y0) < yb )  // could compare sq ?
            {
                 float qa = s.y*s.y+s.z*s.z ; 

                 float qb1 = (o.y-ya)*sy + o.z*s.z ; 
                 float qc1 = (o.y-ya)*(o.y-ya)+o.z*o.z - rr ; 

                 float qb2 = (o.y+ya)*sy + o.z*s.z ; 
                 float qc2 = (o.y+ya)*(o.y+ya)+o.z*o.z - rr ; 

                 float disc1 = qb1*qb1 - qa*qc1 ; 
                 float disc2 = qb2*qb2 - qa*qc2 ; 

                 if( disc1 < 0.f && disc2 < 0.f )
                 {
                      return false ; 
                 } 
            }
        }
*/


    /*
    // try using the rotated ray : see no difference, as expected
 
    const float H = 2.f*RR4*(o.x*s.x+o.y*s.y) ;                  // +/-
    const float G = RR4*(s.x*s.x+s.y*s.y) ;                      // +
    const float I = RR4*(o.x*o.x+o.y*o.y) ;                      // +
    const float J = dot(s, s) ;      // +
    const float K = 2.f*dot(o, s) ;     // +/-
    const float L = dot(o, o) + RR - rr ;  // +    R > r (by assertion)

    */



/*    

                 " residual %10.4f "
                 ,
                 residual
 

                 " neumark_delta2 : %10.3g "
                 ,
                 neumark_delta2


                 " NR ( %10.3g %10.3g %10.3g ) "  


                 " qn( %10.3g, %10.3g, %10.3g, %10.3g)" 
                 " efg( %10.3g, %10.3g, %10.3g )"
                 " neumark( %10.3g, %10.3g, %10.3g )"

                 qn[3],qn[2],qn[1],qn[0]
                 ,
                 e,f,g
                 ,
                 neumark[2],neumark[1],neumark[0]


                 " roots ( %10.3g, %10.3g, %10.3g, %10.3g)" 
                 roots[0], roots[1], roots[2], roots[3]
                 ,
 



                 " HRD %10.4f "
                 HRD
 

                 " R %10.4f r %10.4f "
                  R, r,                      // 1, 0.5


                 " qsd %10.4f "
                 qsd, 
                 qsd    14.5927  

                 " dir (%10.4g %10.4g %10.4g) "
                 ray_direction.x, ray_direction.y, ray_direction.z,

                 dir ( -0.001487 -0.0009288  -0.009845)    ## length of this sq uncomfortably small, following from non-norm for scaling 
                 dir (  -0.07532     0.1403    -0.9872)    ## removing scaling in tboolean-torus makes more healthy


                 " ori (%10.4g %10.4g %10.4g) "
                 ray_origin.x, ray_origin.y, ray_origin.z,

                 ori (    -0.103      2.134     -6.638)     ## as expect


                 " GHIJKL( %10.3g, %10.3g, %10.3g, %10.3g, %10.3g, %10.3g)"     
                 G,H,I,J,K,L

                 GHIJKL(   1.23e-05,    -0.0152,        7.2,     0.0001,     -0.118,       36.1)   ## with scaling of 100
                 GHIJKL(      0.745,      -5.72,       16.9,          1,      -7.62,       16.8)   ## without scaling J=1 norm ray dir
    

                 " q( %10.3g, %10.3g, %10.3g, %10.3g, %10.3g)"    
                 q[4],q[3],q[2],q[1],q[0]     


                 q(      1e-08,  -2.09e-05,     0.0308,      -20.8,   9.79e+03)    ##   v.small q[4] from J**2 from unnorm ray dir
                 qsd   108.6718   q(          1,      -15.2,       90.6,       -249,        263)   ## more reasonable q when avoid scaling


                 " qn( %10.3g, %10.3g, %10.3g, %10.3g)" 
                 qn[3],qn[2],qn[1],qn[0],


                 qsd    52.0278  q(          1,      -14.8,       86.5,       -233,        243) 
                 qn(      -14.8,       86.5,       -233,        243)



*/



