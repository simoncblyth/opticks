
/*

    Ray-Sphere
    ~~~~~~~~~~~~~

    Ray(xyz) = ori + t*dir     dir.dir = 1

    (t*dir + ori-cen).(t*dir + ori-cen) = rad^2

     t^2 dir.dir + 2t(ori-cen).dir + (ori-cen).(ori-cen) - rad^2 = 0  

     t^2 + 2t O.D + O.O - radius = 0 

     a t^2 + b t + c = 0  =>   t = ( -b +- sqrt(b^2 - 4ac) )/2a 


        t = -2 O.D +-  sqrt(4* [(b/2*b/2) - (O.O - rad*rad)])
            ----------------------------------------- 
                            2

          =   - O.D +- sqrt(  O.D*O.D - (O.O - rad*rad) ) 


      normal to sphere at intersection point  (O + t D)/radius

            (ori + t D) - cen
            ------------------
                  radius

 


*/




/*

*intersect_zsphere*
     shooting millions of photons at the +Z pole (Pyrex sphere in vacuum) 
     from a hemi-spherical torch source leads to ~5% level thinking already in
     pyrex when actually in vacuum (ie they failed to intersect at the targetted +Z pole)
     Visible as a red Pyrex leak within the white Vacuum outside the sphere.

     Problem confirmed to be due to bbox effectively clipping the sphere pole
     (and presumably all 6 points where bbox touches the sphere are clipped) 

     Avoid issue by slightly increasing bbox size by factor ~ 1+1e-6


Alternate sphere intersection using geometrical 
rather than algebraic approach, starting from 
t(closest approach) to sphere center

* http://www.vis.uky.edu/~ryang/teaching/cs535-2012spr/Lectures/13-RayTracing-II.pdf



RESOLVED : tangential incidence artifact
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**PILOT ERROR: the sphere was offset at (-1,1,0) when should have been at origin, hangover from leak avoidance?** 

Intersecting a disc of parallel rays of the same radius as a sphere
causes artifacts at tangential incidence (irregular bunched heart shape of reflected photons), 
changing radius of disc to avoid tangentials (eg radius of sphere 100, disc 95) avoids the 
issue.


*/








template<bool use_robust_method>
static __device__
void intersect_zsphere(quad& q0, quad& q1, quad& q2, quad& q3, const uint4& identity)
{
    // intersect z required to be within q2.f.z and q3.f.z (ie the bbox z-range)
    // TODO: move z-range up into (q1.y,q1.z) treating as a zsphere parameters   

    float3 center = make_float3(q0.f);
    float radius = q0.f.w;

/*
    float zmin, zmax ; 
    if(analytic_version > 1)
    {
        zmin = q1.f.y ; 
        zmax = q1.f.z ; 
    }
    else
    {
        zmin = q2.f.z ;
        zmax = q3.f.z ;
    }
*/
    float zmin = q2.f.z ;
    float zmax = q3.f.z ;

    float3 O = ray.origin - center;
    float3 D = ray.direction;

    float b = dot(O, D);
    float c = dot(O, O)-radius*radius;
    float disc = b*b-c;

    /*
    rtPrintf("intersect_zsphere %10.4f %10.4f %10.4f : %10.4f disc %10.4f \n", 
         center.x,  
         center.y,  
         center.z,  
         radius,
         disc);  
    */

    if(disc > 0.0f)
    {
        float sdisc = sqrtf(disc);
        float root1 = (-b - sdisc);
        float root11 = 0.0f;
        bool do_refine = use_robust_method && fabsf(root1) > 10.f * radius ;  // long ray 

        if(do_refine) // refine root1
        {
            float3 O1 = O + root1 * ray.direction;  //  move origin along to 1st intersection point
            b = dot(O1, D);
            c = dot(O1, O1) - radius*radius;
            disc = b*b - c;
            if(disc > 0.0f) 
            {
                sdisc = sqrtf(disc);
                root11 = (-b - sdisc);
            }
        }
        float3 P = ray.origin + (root1 + root11)*ray.direction ;  
        bool check_second = true;

        // require intersection point to be within bbox z range from q2 bbmin and q3 bbmax
        if( P.z >= zmin && P.z <= zmax )
        {
            if( rtPotentialIntersection( root1 + root11 ) ) 
            {
                shading_normal = geometric_normal = (O + (root1 + root11)*D)/radius;
                instanceIdentity = identity ; 
                if(rtReportIntersection(0)) check_second = false;
            } 
        }

        if(check_second) 
        {
            float root2 = (-b + sdisc) + (do_refine ? root11 : 0.f);   // unconfirmed change root1 -> root11
            P = ray.origin + root2*ray.direction ;  
            if( P.z >= zmin && P.z <= zmax )
            { 
                if( rtPotentialIntersection( root2 ) ) 
                {
                    shading_normal = geometric_normal = (O + root2*D)/radius; 
                    instanceIdentity = identity ; 
                    rtReportIntersection(0);   // material index 0 

                    // NB: **NOT** negating normal when inside as that 
                    //     would break rules regards "solidity" of geometric normals
                    //     normal must depend on geometry at intersection point **only**, 
                    //     with no dependence on ray direction
                }
            }
        }
    }
}


