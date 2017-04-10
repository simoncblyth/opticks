
// from tutorial9 intersect_chull
static __device__
void intersect_prism(const uint4& identity)
{
    int nplane = 5 ;

    float t0 = -CUDART_INF_F ; 
    float t1 =  CUDART_INF_F ; 

    float3 t0_normal = make_float3(0.f);
    float3 t1_normal = make_float3(0.f);

    for(int i = 0; i < nplane && t0 < t1 ; ++i ) 
    {
        float4 plane = prismBuffer[i];
        float3 n = make_float3(plane);
        float  d = plane.w;

        float denom = dot(n, ray.direction);
        if(denom == 0.f) continue ;   
        float t = -(d + dot(n, ray.origin))/denom;
    
        // Avoiding infinities.
        // Somehow infinities arising from perpendicular other planes
        // prevent normal incidence plane intersection.
        // This caused a black hairline crack around the prism middle. 
        //
        // In aabb slab method infinities were well behaved and
        // did not change the result, but not here.
        //
        // BUT: still getting extended edge artifact when view from precisely +X+Y
        // http://tavianator.com/fast-branchless-raybounding-box-intersections-part-2-nans/
        //
    
        if( denom < 0.f)  // ray opposite to normal, ie ray from outside entering
        {
            if(t > t0)
            {
                t0 = t;
                t0_normal = n;
            }
        } 
        else 
        {          // ray same hemi as normal, ie ray from inside exiting 
            if(t < t1)
            {
                t1 = t;
                t1_normal = n;
            }
        }
    }

    if(t0 > t1)
        return;

    if(rtPotentialIntersection( t0 ))
    {
        shading_normal = geometric_normal = t0_normal;
        instanceIdentity = identity ;
        rtReportIntersection(0);
    } 
    else if(rtPotentialIntersection( t1 ))
    {
        shading_normal = geometric_normal = t1_normal;
        instanceIdentity = identity ;
        rtReportIntersection(0);
    }
}


/*
Ray-plane intersection 

     n.(O + t D - p) = 0 

     n.O + t n.D - n.p = 0

           t = (n.p - n.O)/n.D  
           t = ( -d - n.O)/n.D
             = -(d + n.O)/n.D

   n.D = 0   when ray direction in plane of face

  why is +Z problematic  n = (0,0,1)

  axial direction rays from +X, +Y, +Z and +X+Y are 
  failing to intersect the prism.  

  Breaking axial with a small delta 0.0001f 
  avoids the issue. 


  See a continuation of edge artifact when viewing from precisely +X+Y


*/






static __device__
float4 make_plane( float3 n, float3 p ) 
{
    n = normalize(n);
    float d = -dot(n, p); 
    return make_float4( n, d );
}

/*

http://mathworld.wolfram.com/Plane.html

    n.(x - p) = 0    normal n = (a,b,c), point in plane p

    n.x - n.p = 0

    ax + by + cz + d = 0        d = -n.p

+Z face of unit cube

    n = (0,0,1)
    p = (0,0,1)
    d = -n.p = -1   ==> z + (-1)  = 0,     z = 1     

-Z face of unit cube

    n = (0,0,-1)
    p = (0,0,-1)
    d = -n.p = -1   ==>  (-z) + (-1) = 0,    z = -1   

*/


static __device__
void make_prism( const float4& param, optix::Aabb* aabb ) 
{
/*
 Mid line of the symmetric prism spanning along z from -depth/2 to depth/2

                                                 
                            A  (0,height,0)     Y
                           /|\                  |
                          / | \                 |
                         /  |  \                +---- X
                        /   h   \              Z  
                       /    |    \ (x,y)   
                      M     |     N   
                     /      |      \
                    L-------O-------R   
         (-hwidth,0, 0)           (hwidth, 0, 0)


    For apex angle 90 degrees, hwidth = height 

*/


    float angle  = param.x > 0.f ? param.x : 90.f ; 
    float height = param.y > 0.f ? param.y : param.w  ;
    float depth  = param.z > 0.f ? param.z : param.w  ;

    float hwidth = height*tan((M_PIf/180.f)*angle/2.0f) ;   

    rtPrintf("make_prism angle %10.4f height %10.4f depth %10.4f hwidth %10.4f \n", angle, height, depth, hwidth);

    float ymax =  height/2.0f ;   
    float ymin = -height/2.0f ;   

    float3 apex = make_float3( 0.f, ymax,  0.f );
    float3 base = make_float3( 0.f, ymin,  0.f) ; 
    float3 front = make_float3(0.f, ymin,  depth/2.f) ; 
    float3 back  = make_float3(0.f, ymin, -depth/2.f) ; 

    prismBuffer[0] = make_plane( make_float3(  height, hwidth,  0.f), apex  ) ;  // +X+Y 
    prismBuffer[1] = make_plane( make_float3( -height, hwidth,  0.f), apex  ) ;  // -X+Y 
    prismBuffer[2] = make_plane( make_float3(     0.f,  -1.0f,  0.f), base  ) ;  //   -Y 
    prismBuffer[3] = make_plane( make_float3(     0.f,    0.f,  1.f), front ) ;  //   +Z
    prismBuffer[4] = make_plane( make_float3(     0.f,    0.f, -1.f), back  ) ;  //   -Z

    rtPrintf("make_prism plane[0] %10.4f %10.4f %10.4f %10.4f \n", prismBuffer[0].x, prismBuffer[0].y, prismBuffer[0].z, prismBuffer[0].w );
    rtPrintf("make_prism plane[1] %10.4f %10.4f %10.4f %10.4f \n", prismBuffer[1].x, prismBuffer[1].y, prismBuffer[1].z, prismBuffer[1].w );
    rtPrintf("make_prism plane[2] %10.4f %10.4f %10.4f %10.4f \n", prismBuffer[2].x, prismBuffer[2].y, prismBuffer[2].z, prismBuffer[2].w );
    rtPrintf("make_prism plane[3] %10.4f %10.4f %10.4f %10.4f \n", prismBuffer[3].x, prismBuffer[3].y, prismBuffer[3].z, prismBuffer[3].w );
    rtPrintf("make_prism plane[4] %10.4f %10.4f %10.4f %10.4f \n", prismBuffer[4].x, prismBuffer[4].y, prismBuffer[4].z, prismBuffer[4].w );

    float3 max = make_float3( hwidth, ymax,  depth/2.f);
    float3 min = make_float3(-hwidth, ymin, -depth/2.f);
    float3 eps = make_float3( 0.001f );

    aabb->include( min - eps, max + eps );

/*
make_prism angle    90.0000 height   200.0000 depth   200.0000 hwidth   200.0000 
make_prism plane[0]     0.7071     0.7071     0.0000  -141.4214 
make_prism plane[1]    -0.7071     0.7071     0.0000  -141.4214 
make_prism plane[2]     0.0000    -1.0000     0.0000    -0.0000 
make_prism plane[3]     0.0000     0.0000     1.0000  -100.0000 
make_prism plane[4]     0.0000     0.0000    -1.0000  -100.0000 
*/

}


