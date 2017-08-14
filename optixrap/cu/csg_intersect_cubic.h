using namespace optix;

#define RRZ(z) (((A*(z)+B)*(z) + C)*(z) + D)

static __device__
void csg_bounds_cubic(const quad& q0, const quad& q1,  optix::Aabb* aabb, optix::Matrix4x4* tr  )
{
    const float& A = q0.f.x ;
    const float& B = q0.f.y ;
    const float& C = q0.f.z ;
    const float& D = q0.f.w ;

    const float& z1 = q1.f.x ;
    const float& z2 = q1.f.y ;
 
/*

Maximum radius^2 within z range, either from local max "bulge" that is the domain max 
or z1/z2 endpoint max.

* x^2 + y^2  = rr =  A z^3 + B z^2 + C z + D   

*  d(rr)/dz = 3 A z^2 + 2 B z + C 

Local minimum or maximum at 2 z values, from quadratic roots of derivative

   -B +- sqrt( B^2 - 3 A C )
  --------------------------
        3 A
*/

    float d = 3.f*A ; 

    float disc = B*B - d*C ; 
    float sdisc = disc > 0.f ? sqrtf(disc) : 0.f ; 

    float q = B > 0.f ? -(B + sdisc) : -(B - sdisc) ;
    float e1 = q/d  ;
    float e2 = C/q  ;

    float4 rr = make_float4(  
                    RRZ(z1), 
                    RRZ(z2), 
                    disc > 0.f && e1 > z1 && e1 < z2 ? RRZ(e1) : 0.f , 
                    disc > 0.f && e2 > z1 && e2 < z2 ? RRZ(e2) : 0.f 
               ); 
  
    float rrmx = fmaxf(rr) ;
    float rmx = sqrtf(rrmx);

    rtPrintf("// csg_bounds_cubic ABCD (%g %g %g %g)  z1 z2 (%g %g)  rr(%g %g %g %g)     \n", A,B,C,D,z1,z2,rr.x,rr.y,rr.z,rr.w );
 
    float3 mn = make_float3(  -rmx,  -rmx,  z1 );
    float3 mx = make_float3(   rmx,   rmx,  z2 );

    Aabb tbb(mn, mx);
    if(tr) transform_bbox( &tbb, tr );  

    aabb->include(tbb);
}


static __device__
bool csg_intersect_cubic(const quad& q0, const quad& q1, const float& t_min, float4& isect, const float3& ray_origin, const float3& ray_direction )
{
   /*

      x^2 +  y^2  - A*z^3 - B*z^2 - C*z - D  = 0 


      from sympy import symbols, expand, Poly

      A,B,C,D,ox,oy,oz,sx,sy,sz,t,x,y,z = symbols("A,B,C,D,ox,oy,oz,sx,sy,sz,t,x,y,z")

      f = x*x + y*y - A*z*z*z - B*z*z - C*z - D 

      ff =  f.subs([[x,ox+t*sx],[y,oy+t*sy],[z,oz+t*sz]])

        
    In [112]: Poly(ff, t).all_coeffs()
    Out[112]: 
    [-A*sz**3,
     -3*A*oz*sz**2 - B*sz**2 + sx**2 + sy**2,
     -3*A*oz**2*sz - 2*B*oz*sz - C*sz + 2*ox*sx + 2*oy*sy,
     -A*oz**3 - B*oz**2 - C*oz - D + ox**2 + oy**2]

 

     grad( x^2 + y^2 - A * z^3 - B * z^2 - C*z - D ) =  [2 x, 2 y, -3 A z^2 - 2 B z - C ] 




      x^2 +  y^2  - A*z^3 - B*z^2 - C*z - D  = 0 


     Consider changing the unit of length with the aim of bringing typical lengths
     into ballpark of the unit cube, for numerical stability/precision reasons.
     Choosing a unit length s=z2-z1 characteristic of the geometry.

     Magic of parametric-t means there is no impact on it.

     For example say z = 200,  with z1=-200, z2=200, s=z2-z1 = 400
                     z = 200 = 400*0.5
                     z = 200 = s*zs      where zs is scaled-z

     Changing length unit corresponds for factoring off a constant from all lengths,
     which will mean scaling coeefs ...

      (xs*s)^2 +  (ys*s)^2  - A*(zs*s)^3 - B*(zs*s)^2 - C*(zs*s) - D  = 0 

       xs^2 + ys^2 - A*s*zs^3 - B*zs^2 - (C/s)*zs - D/s^2 = 0     // dividing by s^2

    
       A -> s*A
       B ->  B
       C -> C*is
       D -> D*is*is


    :google:`polynomial scaling coeff numerical`

    * https://math.stackexchange.com/questions/1384741/how-to-scale-polynomial-coefficients-for-root-finding-algorithms
    * http://www.akiti.ca/rpoly_ak1_cpp.html

   */

    const Cubic_t zero(0); 
    const Cubic_t one(1); 
    const Cubic_t two(2); 
    const Cubic_t three(3); 
    const Cubic_t four(4); 

    const Cubic_t s = q1.f.y - q1.f.x ; // unscaled z2 - z1,  z2 > z1 by assertion, s +ve
    //const Cubic_t s = one ; 
    const Cubic_t is = one/s ;
 
    const Cubic_t z1 = is*q1.f.x ; 
    const Cubic_t z2 = is*q1.f.y ;
   
    const Cubic_t sx = is*ray_direction.x ; 
    const Cubic_t sy = is*ray_direction.y ; 
    const Cubic_t sz = is*ray_direction.z ;

    const Cubic_t ox = is*ray_origin.x ; 
    const Cubic_t oy = is*ray_origin.y ; 
    const Cubic_t oz = is*ray_origin.z ;

    // attempt to adjust cubic coeff to effect the unit length scaling 

    const Cubic_t A = q0.f.x*s ;
    const Cubic_t B = q0.f.y ; 
    const Cubic_t C = q0.f.z*is ; 
    const Cubic_t D = q0.f.w*is*is ;  

    const Cubic_t a = -A*sz*sz*sz ; 
    const Cubic_t b = -three*A*oz*sz*sz - B*sz*sz + sx*sx + sy*sy ; 
    const Cubic_t c = -three*A*oz*oz*sz - two*B*oz*sz - C*sz + two*ox*sx + two*oy*sy ; 
    const Cubic_t d = -A*oz*oz*oz - B*oz*oz - C*oz - D + ox*ox + oy*oy ; 


    Cubic_t trev[3] ; 
    float zrev[3] ; 
    unsigned nr ; 

    const Cubic_t acut = 1e-3 ; 

    // artifact blow out for sz->0 (ie rays within xy plane, edge on view)
    //  a->0, coeff ->infinity 
    // cubic term disappears, degenerating into quadratic 

    if( fabs(a) < acut )
    {
        const Cubic_t q = c/b ; 
        const Cubic_t r = d/b ; 
        const Cubic_t disc = q*q - four*r ;  
        nr = SolveQuadratic( q,  r , trev, disc, zero ) ; 
    }
    else
    {
        const Cubic_t p = b/a ; 
        const Cubic_t q = c/a ; 
        const Cubic_t r = d/a ; 
     
        unsigned msk = 0u ; 
        nr = SolveCubic( p, q, r, trev, msk );
    }

    zrev[0] = nr > 0 && trev[0] > t_min ? oz + trev[0]*sz : RT_DEFAULT_MAX ; 
    zrev[1] = nr > 1 && trev[1] > t_min ? oz + trev[1]*sz : RT_DEFAULT_MAX ; 
    zrev[2] = nr > 2 && trev[2] > t_min ? oz + trev[2]*sz : RT_DEFAULT_MAX ; 


    //  z = oz+t*sz -> t = (z - oz)/sz 
    const float osz = one/sz ; 
    const float t2cap = (z2 - oz)*osz ;   // cap plane intersects
    const float t1cap = (z1 - oz)*osz ;

    const float c1x = ox + t1cap*sx ; 
    const float c1y = oy + t1cap*sy ; 
    const float c2x = ox + t2cap*sx ; 
    const float c2y = oy + t2cap*sy ; 

    float crr1 = c1x*c1x + c1y*c1y ;   // radii squared at cap plane intersects
    float crr2 = c2x*c2x + c2y*c2y ; 

    // cap planes shape-of-revolution radii^2
    const float rr1 = RRZ(z1) ; 
    const float rr2 = RRZ(z2) ; 
 
    // NB must disqualify t < t_min at "front" and "back" 
    // as this potentially picks between intersects eg whilst near(t_min) scanning  
    //
    // restrict radii of cap intersects and z of rev intersects

    float tcan[5] ;  

    tcan[0] = t2cap > t_min && crr2 < rr2  ? t2cap   : RT_DEFAULT_MAX ;
    tcan[1] = t1cap > t_min && crr1 < rr1  ? t1cap   : RT_DEFAULT_MAX ;
    tcan[2] = zrev[0] > z1 && zrev[0] < z2 ? trev[0] : RT_DEFAULT_MAX ; 
    tcan[3] = zrev[1] > z1 && zrev[1] < z2 ? trev[1] : RT_DEFAULT_MAX ; 
    tcan[4] = zrev[2] > z1 && zrev[2] < z2 ? trev[2] : RT_DEFAULT_MAX ; 
   
    float t_cand = RT_DEFAULT_MAX  ; 
    for(unsigned i=0 ; i < 5 ; i++ ) if( tcan[i] < t_cand ) t_cand = tcan[i] ; 
 

    bool valid_isect = t_cand > t_min && t_cand < RT_DEFAULT_MAX ;
    if(valid_isect)
    {        
        isect.w = t_cand ; 
        if( t_cand == tcan[2] || t_cand == tcan[3] || t_cand == tcan[4] )
        {
            const float px = ox + t_cand*sx ; 
            const float py = oy + t_cand*sy ; 
            const float pz = oz + t_cand*sz ; 

            // grad( x^2 + y^2 - A * z^3 - B * z^2 - C*z - D ) =  [2 x, 2 y, -3 A z^2 - 2 B z - C ] 

            float3 n = normalize(make_float3( 2.f*px,  2.f*py,  -3.f*A*pz*pz -2.f*B*pz - C  )) ;   
            isect.x = n.x ; 
            isect.y = n.y ; 
            isect.z = n.z ;      
        }
        else
        {
            isect.x = zero ; 
            isect.y = zero ; 
            isect.z = t_cand == t1cap ? -one : one ;  
        }
    }
    return valid_isect ; 
}


