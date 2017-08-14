
#define TORUS_DEBUG 1


/*
typedef double Solve_t ;
typedef double Torus_t ;

#include "Solve.h"

*/


using namespace optix;

static __device__
void csg_bounds_torus(const quad& q0, optix::Aabb* aabb, optix::Matrix4x4* tr  )
{
    const float rminor = q0.f.z ;     
    const float rmajor = q0.f.w ;    
    const float rsum = rminor + rmajor ;  

    float3 mn = make_float3( -rsum, -rsum,  -rminor );
    float3 mx = make_float3(  rsum,  rsum,   rminor );

    Aabb tbb(mn, mx);
    if(tr) transform_bbox( &tbb, tr );  

    aabb->include(tbb);


//#ifdef CSG_INTERSECT_TORUS_TEST
    rtPrintf("// csg_bounds_torus rmajor %f rminor %f rsum %f  tr %d  \n", rmajor, rminor, rsum, !!tr );
//#endif




}




static __device__
float cubic_delta2( const float a, const float b, const float c, const float d)
{
    // Neumark p5  :   a x^3 + b x^2 + c x + d = 0 
    //
    float tmp = 27.f*a*a*d + 2.f*b*b*b - 9.f*a*b*c ; 
    float disc = b*b - 3.f*a*c ; 
    float delta2 = tmp*tmp / (2.f*disc*disc*disc) ; 
    return delta2 ;
}

static __device__
void z_rotate_ray_align_x(const float3& o0, const float3& s0, float3& o, float3& s)
{ 
    const float phi = atan2(s0.x, s0.y) ;

    //s.x = s0.x*cos(phi) - s0.y*sin(phi) ;  
    s.x = 0.f ;                           // <-- by construction the rotation aims to make this very small 
    s.y = s0.y*cos(phi) + s0.x*sin(phi) ;
    s.z = s0.z ; 

    o.x = o0.x*cos(phi) - o0.y*sin(phi) ;
    o.y = o0.y*cos(phi) + o0.x*sin(phi) ;
    o.z = o0.z ; 
}

/*
     

Normalization
---------------

In principal its better not to normalize ray_direction 
in order to support non-uniform scaling, 
but in practice even moderate scalings such as 100x typical whilst testing
result in numerical issues from very small ray_direction.
   
Unnormalized using source length unit (mm) and whatever ray_direction
length that any scaling causes::

   ray(per-mm) = ray_origin + t * ray_direction 
   ray(per-R)  = ray_origin/R_ + (t * ray_direction)/R_ 

Changing length unit such that torus R=1. effects all terms, so 
without amending 


  
To handle non-uniform scaling would need to factor
scaling into largest common denom uniform 
scaling with smaller variation on top of that.
    
Normalization by the length of ray_direction assumes uniform scaling 
    
Usually have t equivalence between frames as do not have numerical
domain issues and can just live with scaled down qtys.
     
Scalings and R-norming are sorta doing same 
thing that makes them confusing 
s is the unit of t 

The reason for scaling in first place was to allow 
use of R ~ 1 : but then you get bitten in butt by 
small sx,sy,sz,ox,oy,oz

*/



static __device__
bool csg_intersect_torus(const quad& q0, const float& t_min, float4& isect, const float3& ray_origin, const float3& ray_direction )
{
    const Torus_t zero(0) ; 
    const Torus_t one(1) ; 
    const Torus_t two(2) ; 
    const Torus_t four(4) ; 

    const Torus_t R_ = q0.f.w ;  
    const Torus_t r_ = q0.f.z ;  // R_ > r_ by assertion, so torus has a hole   

    const Torus_t ss = dot( ray_direction, ray_direction );
    const Torus_t unit = sqrt(ss);   
    //const Torus_t unit = R_ ;   

    const Torus_t sx = ray_direction.x/unit ;
    const Torus_t sy = ray_direction.y/unit ;  
    const Torus_t sz = ray_direction.z/unit ;

    const Torus_t ox = ray_origin.x/unit ; 
    const Torus_t oy = ray_origin.y/unit ; 
    const Torus_t oz = ray_origin.z/unit ; 

    const Torus_t R = R_/unit ; 
    const Torus_t r = r_/unit ; 

    // Note the parametric ray trace magic : so long as all
    // lengths are scaled the same there is no impact on the parametric t
    // as t is defined as a multiple of the ray direction vector
    //
    // scaled ray dir, ori too close to origin for numerical comfort
    // due to scale factors to enable use of small R_ r_ 
    // so divide by unit to bring into viscinity of unity 
    // but must treat all lengths same ... so the radii get blown up ???
    // and upshot is the coeffs come out the same ???
    //
    // Need to check quartic coeff disparity to see what approach is best
   
    
#ifdef CSG_INTERSECT_TORUS_TEST
    rtPrintf("// csg_intersect_torus R r unit (%g %g %g)  oxyz (%g %g %g) sxyz (%g %g %g ) t_min (%g)   \n", 
              R, r, unit
              ,
              ox,oy,oz
              ,
              sx,sy,sz
              ,
              t_min
             ); 

#endif      

    const Torus_t rmax = R+r ; 
    const Torus_t rmin = R-r ; 
 
    const Torus_t rr = r*r ; 
    const Torus_t RR = R*R ; 
    const Torus_t RR4 = RR*four ; 

    /*
       Closest approach of ray (r = o + s t) to torus-axis (z), 
       from turning point minimization xy radial distance squared

            rxysq =  (ox + sx t)^2 + (oy + sy t)^2  

                  = (ox^2 + oy^2) + 2t ( ox sx + oy sy ) + (sx^2 + sy^2) t^2  
    
      d(rxysq)/dt =    2 (ox sx + oy sy ) + 2t (sx^2 + sy^2 ) 
         
              -> tc = - (ox sx + oy sy)     at closest point 
                        ---------------- 
                         sx sx + sy sy 
    */

    const Torus_t oxox_oyoy = ox*ox + oy*oy ; 
    const Torus_t oxsx_oysy = ox*sx + oy*sy ; 
    const Torus_t sxsx_sysy = sx*sx + sy*sy ; 

    const Torus_t tc = -oxsx_oysy/sxsx_sysy ;    
    const Torus_t xc = ox + sx*tc ; 
    const Torus_t yc = oy + sy*tc ; 
    const Torus_t zc = oz + sz*tc ; 
    const Torus_t rcrc = xc*xc + yc*yc ;   // square of distance to axis at closest approach 

    const Torus_t rmax_rmax = rmax*rmax ; 
    const Torus_t rmin_rmin = rmin*rmin ; 


    if( rcrc > rmax_rmax )   // intersect not possible when closest approach to axis exceeds rmax
    {
        /*
        rtPrintf("  R r rmax rmin (%g %g %g %g) xc yc zc (%g %g %g)  rcrc rmax_rmax (%g, %g)   \n",
                       R,r,rmax,rmin, xc,yc,zc, rcrc, rmax_rmax   ); 
        */

        return false ; 
    }

    // following cosinekitty nomenclature, sympy verified in torus.py
  
    const Torus_t H = two*RR4*(oxsx_oysy) ;         // +/-
    const Torus_t G = RR4*(sxsx_sysy) ;             // +
    const Torus_t I = RR4*(oxox_oyoy) ;             // +
    const Torus_t J = sxsx_sysy + sz*sz  ;          // +
    const Torus_t K = two*(oxsx_oysy + oz*sz) ;     // +/-
    const Torus_t L = oxox_oyoy + oz*oz + RR - rr ; // +    R > r (by assertion)
   
    // A x**4 + B x**3 + C x**2 + D x + E = 0 
    // E (1/x)**4 + D (1/x)^3 + C (1/x)^2 + B (1/x) + A = 0     divide by x**4  


    const Torus_t A = J*J ;                                   // +
    const Torus_t B = two*J*K ;                               // +/-
    const Torus_t C = two*J*L + K*K - G ;                     // +/-
    const Torus_t D = two*K*L - H ;                           // +/-
    const Torus_t E = L*L - I ;                               // +/-

/*
    const Torus_t AOB = fabs(A/B);
    const Torus_t EOD = fabs(E/D);
    bool reverse = (EOD + one/EOD) > (AOB + one/AOB) ;     
*/
    bool reverse = false ; 

    Torus_t qn[4] ; 
    qn[3] = reverse ? D/E : B/A ;
    qn[2] = reverse ? C/E : C/A ;
    qn[1] = reverse ? B/E : D/A ;
    qn[0] = reverse ? A/E : E/A ;


#ifdef CSG_INTERSECT_TORUS_TEST
    rtPrintf("// csg_intersect_torus HGIJKL (%g %g %g %g %g %g)  ABCDE (%g %g %g %g %g ) \n", 
              H,G,I,J,K,L
              ,
              A,B,C,D,E             
             ); 

    rtPrintf("// csg_intersect_torus qn (%g %g %g %g) reverse %d \n", 
              qn[3],qn[2],qn[1],qn[0], reverse
             ); 

#endif      



   // unsigned msk = SOLVE_VECGEOM  ;  // worst for artifcacting 
   // unsigned msk = SOLVE_UNOBFUSCATED  ;  // reduced but still obvious
   // unsigned msk = SOLVE_UNOBFUSCATED | SOLVE_ROBUSTQUAD ;  // getting some in-out wierdness
   // unsigned msk = SOLVE_UNOBFUSCATED | SOLVE_ROBUSTQUAD | SOLVE_ROBUSTCUBIC_0 ;
   unsigned msk = SOLVE_UNOBFUSCATED | SOLVE_ROBUSTCUBIC_0 | SOLVE_ROBUSTCUBIC_1 | SOLVE_ROBUSTCUBIC_2 | SOLVE_ROBUSTQUAD_1 | SOLVE_ROBUST_VIETA  ;  // _0 ok
   // unsigned msk = SOLVE_UNOBFUSCATED | SOLVE_ROBUSTCUBIC_1 ;  // in-out wierdness
 

    Solve_t roots[4] ;   
    //int num_roots = SolveQuarticPureNeumark( qn[3],qn[2],qn[1],qn[0], roots, msk ); 
    int num_roots = SolveQuartic( qn[3],qn[2],qn[1],qn[0], roots, msk ); 



    float4 cand = make_float4(RT_DEFAULT_MAX) ;  
    int num_cand = 0 ;  
    if(num_roots > 0)
    {
        for(int i=0 ; i < num_roots ; i++)
        {
            Solve_t root = reverse ? one/roots[i] : roots[i] ; 
            if(root > t_min ) setByIndex(cand, num_cand++, root ) ;   // back to floats
        }   
    }
    
    float t_cand = num_cand > 0 ? fminf(cand) : t_min ;   // smallest root bigger than t_min

    const float3 p0 = make_float3( ox + t_cand*sx, oy + t_cand*sy, oz + t_cand*sz )  ;   


#ifdef CSG_INTERSECT_TORUS_TEST
    rtPrintf(" t_cand %g  p0 (%g %g %g) \n", 
                t_cand
                ,
                p0.x, p0.y, p0.z 
            );
#endif      

    //const float pr = sqrt(p0.x*p0.x+p0.y*p0.y) ;   // <-- selecting artifact in hole 
    //const float2 qrz = make_float2(length(make_float2(p0)) - R, p0.z) ;  // (mid-circle-sdf, z)
    //const float  qsd = length(qrz) - r ;   // signed dist to torus
    //const float  aqsd = fabsf(qsd) ;      // dist to torus
    //bool valid_qsd = aqsd < 1e-3f ; 


    // Can easily cut away fake intersects assumed caused by numerical problems 
    // for some coeffient combinations using sdf... 
    // but how to handle failed intersects ?
    //
    // Inverting !valid_qsd to see just the fakes
    // suggests interior missing intersects and
    // exterior fake intersects arise from same cause.
    //
    // So identifying cause of exterior fakes (eg some coeff going to zero)
    // may potentially also fix interior missings.

    bool valid_isect = t_cand > t_min ;    

    if(valid_isect)
    {        
        const float alpha = 1.f - (R/sqrt(p0.x*p0.x+p0.y*p0.y)) ;   // see cosinekitty 
        const float3 n = normalize(make_float3(alpha*p0.x, alpha*p0.y, p0.z ));
        isect.x = n.x ;  
        isect.y = n.y ;  
        isect.z = n.z ;  
        isect.w = t_cand ; 
    }
    return valid_isect ; 
}


#ifdef CSG_INTERSECT_TORUS_TEST
static __device__
void dump_TVQ(const Matrix4x4& T, const Matrix4x4& V, const Matrix4x4& Q)
{
    rtPrintf("// T(transform)\n%8.3f %8.3f %8.3f %8.3f\n%8.3f %8.3f %8.3f %8.3f\n%8.3f %8.3f %8.3f %8.3f\n", 
          T[0], T[1], T[2], T[3],  
          T[4], T[5], T[6], T[7],  
          T[8], T[9], T[10], T[11]
         );  
    rtPrintf("%8.3f %8.3f %8.3f %8.3f\n", T[12], T[13], T[14], T[15] );

    rtPrintf("// V(inverse)\n%8.3f %8.3f %8.3f %8.3f\n%8.3f %8.3f %8.3f %8.3f\n%8.3f %8.3f %8.3f %8.3f\n", 
          V[0], V[1], V[2], V[3],  
          V[4], V[5], V[6], V[7],  
          V[8], V[9], V[10], V[11]
         );  
    rtPrintf("%8.3f %8.3f %8.3f %8.3f\n", V[12], V[13], V[14], V[15] );

    rtPrintf("// Q(inverse-transposed)\n%8.3f %8.3f %8.3f %8.3f\n%8.3f %8.3f %8.3f %8.3f\n%8.3f %8.3f %8.3f %8.3f\n", 
          Q[0], Q[1], Q[2], Q[3],  
          Q[4], Q[5], Q[6], Q[7],  
          Q[8], Q[9], Q[10], Q[11]
         );  
    rtPrintf("%8.3f %8.3f %8.3f %8.3f\n", Q[12], Q[13], Q[14], Q[15] );

}
#endif

 


static __device__
void csg_intersect_torus_scale_test(unsigned long long photon_id, bool do_scale)
{
   // check in plane of torus, for normalization/scaling correctness

    const float r = 10.f ; 
    const float R = 100.f ; 
    const float rmax = R + r  ;  
    const float rmin = R - r ; 

/*
   // SCALE TEST
    float3 ori = make_float3( -rmax - r , 0.f, 0.f );  
    float3 dir = make_float3( 1.f       , 0.f, 0.f );       
    const float4  expect_ = make_float4( -rmax, -rmin, rmin, rmax );  
 
*/
   // HOLE TEST

    // With delta 0. this produces part of the in-the-hole artifact ring in perspective proj, line in ortho.
    // So what is different between the ring and those to the side that 
    // correctly return no isect ?
    //const float delta = 10.f ; 

    const float delta = 0.f ; 
    float3 ori = make_float3(     -64.6f+delta,    0.5311f,     394.7f); 
    float3 dir = make_float3(     0.059f,   0.07738f,   -0.9953f);

    //float3 ori = make_float3(     -64.6f,    0.5311f,     394.7f); 
    //float3 dir = make_float3(     0.f,   0.f,   -1.f);

    Ray ray = make_Ray( ori, dir, 0u, 0.f, RT_DEFAULT_MAX ); 

    const float uscale = do_scale ? R : 1.f  ;  

    //////////// "world" frame above : scaling is implementation detail /////////////
    const float r_ = r/uscale ; 
    const float R_ = R/uscale ;     
    const float3 scale_ = make_float3( uscale, uscale, uscale );

    quad q0 ; 
    q0.f.z = r_ ; 
    q0.f.w = R_ ; 
 
    Matrix4x4 T = Matrix4x4::scale(scale_) ; 
    Matrix4x4 V = T.inverse() ;
    Matrix4x4 Q = V.transpose() ;


#ifdef CSG_INTERSECT_TORUS_TEST
    rtPrintf("\n\n// csg_intersect_torus_scale_test uscale %g \n", uscale );
    dump_TVQ(T,V,Q); 
#endif
   
    // suspect matrix layout from these ctors is transposed
    // wrt Opticks standard (following OpenGL) ... but 
    // it does not matter for scaling, and use of these
    // ctors is just for testing anyhow as it clearly
    // is preferable to do such things once only CPU side.

  
    float4 origin    = make_float4( ray.origin.x,    ray.origin.y,    ray.origin.z,    1.f );  // w=1 for position  
    float4 direction = make_float4( ray.direction.x, ray.direction.y, ray.direction.z, 0.f );  // w=0 for direction

    origin    = origin * V ;     // world frame into primitive frame with inverse transform
    direction = direction * V ;  // <-- will loose normalization with scaling, intersects MUST NOT assume normalized ray direction

    float3 ray_origin    = make_float3( origin.x, origin.y, origin.z );
    float3 ray_direction = make_float3( direction.x, direction.y, direction.z ); 

    const float3& o = ray_origin ;
    const float3& d = ray_direction ;


#ifdef CSG_INTERSECT_TORUS_TEST
    rtPrintf("// pid %llu \n", photon_id );
    rtPrintf("// csg_intersect_torus_test  r R rmax (%g %g %g) ray_origin (%g %g %g) ray_direction (%g %g %g) \n"
              ,
              r,R,rmax
              ,
              o.x,o.y,o.z
              ,
              d.x,d.y,d.z
            );
#endif


    float t_min = 0.f ; 
    float4 tt = make_float4(0.f,0.f,0.f,0.f);

    for(unsigned i=0 ; i < 1 ; i++)
    {
        bool valid_isect = csg_intersect_torus(q0, t_min , tt, o, d );

        if(!valid_isect) 
        {
#ifdef CSG_INTERSECT_TORUS_TEST
            rtPrintf("ERROR no isect \n");
#endif 
            break ; 
        } 


        float4 ttn = make_float4( tt.x, tt.y, tt.z , 0.f );
        ttn = ttn * Q   ;  // primitive frame normal into world frame, using inverse transform transposed

        tt.x = ttn.x ; 
        tt.y = ttn.y ; 
        tt.z = ttn.z ; 
            
        float t = tt.w ;   //  Ray trace magic keeps t the same both with and without scaling  
        float3 p = ray.origin + t*ray.direction ; 



/*
        float expect = getByIndex(expect_, i ) ; 
        if(fabsf( p.x - expect) > 2.f )
        {
            rtPrintf("ERROR x expect deviation p.x %g expect %g  t %g  \n",  p.x, expect, t );
            break ;  
        }
*/

#ifdef CSG_INTERSECT_TORUS_TEST
        rtPrintf("// csg_intersect_torus_test t_min %10.4g    tt:(%10.3f %10.3f %10.3f %10.3f) p:(%10.3f %10.3f %10.3f) \n",
                       t_min, tt.x, tt.y, tt.z, tt.w, p.x, p.y, p.z ); 

#endif 

        t_min = t + 1e-4f  ; 

   }

}


