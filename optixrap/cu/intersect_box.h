/*

Ray Box Intersection Slab Method
==================================

* http://tavianator.com/fast-branchless-raybounding-box-intersections/


                   |            |  / 
                   |            | /
                   |            |/ 
                   |            $ 
                   |           /|
                   |          / |
                   |         /  |
                   |        /   |
                   |       /    |
                   |      /     |
          .        |     /      |
          .        |    /       |
          .        |   /        | 
          .        |  /         |
   +      +--------+-$:B--------X----------- bbmax y plane
   |      .        |/           |
   |      .      A:@            |
   |      .       /|     C      |
   |      .      / |            |
 t1.y     .     /  |            |
   |  +   +----@---N------------+------------ bbmin y plane
   |  |   .   /    |            |
   | t0.y .  /     |            |
   |  |   . /      |            |
   |  |   ./       |            |
...+..+.+.O........+............+.........
         /.                      
        / .                      
       /  +--t0.x--+             
          .                      
          +---------t1.x--------+
                                 
          +-near.x-+

          +---------far.x-------+



         O : ray.origin    (t = 0)

         X : bb.max

         N : bb.min

         C : bb.center (min+max)/2

         @ : intersections with the bbmin planes

         $ : intersections with the bbmax planes



      near : min(t0, t1)   [ min(t0.x, t1.x), min(t0.y,t1.y), min(t0.z,t1.z) ]
      far  : max(t0, t1)   [ max(t0.x, t1.x), max(t0.y,t1.y), max(t0.z,t1.z) ] 

             ordering of slab intersections for each axis, into nearest and furthest 

             ie: pick either bbmin or bbmax planes for each axis 
             depending on the direction the ray is coming from

             think about looking at a box from multiple viewpoints 
             

      tmin : max(near)

             pick near slab intersection with the largest t value, 
             ie furthest along the ray, so this picks "A:@" above
             rather than the "@" slab(not-box) intersection

      tmax : min(far)

             pick far slab intersection with the smallest t value,
             ie least far along the ray, so this picks "$:B" above
             rather than the "$" slab(not-box) intersection 

      tmin <= tmax 

             means the ray has a segment within the box, ie intersects with it, 
             but need to consider different cases:


      tmin <= tmax && tmin > 0 

                       |        |
              ---O->---+--------+----- 
                       |        | 
                     tmin     tmax 

             ray.origin outside the box
             
             intersection at t = tmin


      tmin <= tmax && tmin < 0 

                       |        |
                 ------+---O->--+----- 
                       |        | 
                     tmin     tmax 

             ray.origin inside the box, so there is intersection in direction
             opposite to ray.direction (behind the viewpoint) 

             intersection at t = tmax


      tmin <= tmax && tmax < 0    (implies tmin < 0 too)

                       |        |
                 ------+--------+---O->-- 
                       |        | 
                     tmin     tmax 


             ray.origin outside the box, with intersection "behind" the ray 
             so must disqualify the intersection

      
       tmin <= tmax && tmax > 0 
 
             qualifying intersection condition, with intersection at 

                    tmin > 0 ? tmin : tmax

             where tmin < 0 means are inside



       is check_second needed ?

             YES for rendering (not for propagation, but does no harm?) 
             this handles OptiX epsilon near clipping,
             a way to clip ray trace rendered geometry so can look inside
             in this situation valid front face box intersections 
             will be disqualifies so need to fall back to the other intersection


       Normals at intersections will be in one of six directions: +x -x +y -y +z -z 

             http://graphics.ucsd.edu/courses/cse168_s06/ucsd/CSE168_raytrace.pdf

       Consider vector from box center to intersection point 
       ie intersect coordinates in local box frame
       Think of viewing unit cube at origin from different
       directions (eg face on down the axes).

       Determine which face from the largest absolute 
       value of (x,y,z) of local box frame intersection point. 

       Normal is defined to be in opposite direction
       to the impinging ray. 

       * **WRONG : NOT AT INTERSECT LEVEL
       * INTERSECT LEVEL IS FOR DEFINING GEOMETRY, NOT THE RELATIONSHIP 
         OF VIEWPOINT AND GEOEMTRY : THAT COMES LATER
          
 
                  +---@4------+
                  |   /\      |
                  |   +       |
             +->  @1     +->  @2  
                  |   +       |
                  |   \/      @3   <-+
                  +---@5--@6--+
                          /\
                          + 

                               normal

            @1   [-1,0,0]    =>  [-1,0,0]   ( -x from outside, no-flip )     

            @2   [ 1,0,0]        [-1,0,0]   ( +x from inside, flipped ) 

            @3   [ 1,-0.7,0] =>  [ 1,0,0]   ( +x from outside, no-flip ) 

            @4   [-0.5,1,0]  =>  [ 0,-1,0]   ( -y from inside, flipped )

            @5   [-0.5,-1,0] =>  [ 0, 1,0]   ( +y from inside, flipped )

            @6   [ 0.5,-1,0] =>  [ 0,-1,0]   ( -y from outside, no-flip)



      RULES FOR GEOMETRIC NORMALS  
      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      * MUST BE RIGIDLY AND CONSISTENTLY "ATTACHED" TO GEOMETRY DEPENDING ONLY ON
        THE GEOMETRY AT THE INTERSECTION POINT AND SOME FIXED CONVENTION 
        (CONSIDER TRIANGLE INTERSECTION EXAMPLE)

      * I ADOPT STANDARD CONVENTION OF OUTWARDS POINTED NORMALS : 
        SO IT SHOULD BE DARK INSIDE BOXES

      THIS MEANS:

      **DO NOT FLIP BASED ON WHERE THE RAYS ARE COMING FROM OR BEING INSIDE BOX**   


      Edge Cases
      ~~~~~~~~~~~~

      http://tavianator.com/fast-branchless-raybounding-box-intersections/
      http://tavianator.com/fast-branchless-raybounding-box-intersections-part-2-nans/ 
 
      Floating-point infinity handling should correctly deal 
      with axis aligned ray directions but there is literally an edge case 
      when the ray starts exactly on the edge the box 
    
          0 * inf = nan which occurs when the ray starts 
  




*/

static __device__
void intersect_aabb(quad& q2, quad& q3, const uint4& identity)
{
    // using q2 and q3 for bbox

    const float3 min_ = make_float3(q2.f.x, q2.f.y, q2.f.z); 
    const float3 max_ = make_float3(q3.f.x, q3.f.y, q3.f.z); 

    const float3 cen_ = 0.5f*(min_ + max_) ;    

    float3 t0 = (min_ - ray.origin)/ray.direction;
    float3 t1 = (max_ - ray.origin)/ray.direction;

    // slab method 
    float3 near = fminf(t0, t1);
    float3 far = fmaxf(t0, t1);
    float tmin = fmaxf( near );
    float tmax = fminf( far );


    if(tmin <= tmax && tmax > 0.f) 
    {
        bool check_second = true;
        float tint = tmin > 0.f ? tmin : tmax ; 

        if(rtPotentialIntersection(tint))
        {
            float3 p = ray.origin + tint*ray.direction - cen_ ; 
            float3 pa = make_float3(fabs(p.x), fabs(p.y), fabs(p.z)) ;
            float pmax = fmaxf(pa);

            float3 n = make_float3(0.f) ;
            if(      pa.x >= pa.y && pa.x >= pa.z ) n.x = copysignf( 1.f , p.x ) ;              
            else if( pa.y >= pa.x && pa.y >= pa.z ) n.y = copysignf( 1.f , p.y ) ;              
            else if( pa.z >= pa.x && pa.z >= pa.y ) n.z = copysignf( 1.f , p.z ) ;              


            shading_normal = geometric_normal = n ;
            instanceIdentity = identity ;
            if(rtReportIntersection(0)) check_second = false ;   // material index 0 
        } 

        // handle when inside box, or are epsilon near clipped 
        if(check_second)
        {
            if(rtPotentialIntersection(tmax))
            {
                float3 p = ray.origin + tmax*ray.direction - cen_ ; 
                float3 pa = make_float3(fabs(p.x), fabs(p.y), fabs(p.z)) ;
                float pmax = fmaxf(pa);

                /*
                float3 n = make_float3(0.f);  

                if(      pa.x >= pa.y && pa.x >= pa.z ) n.x = copysignf( 1.f , p.x ) ;              
                else if( pa.y >= pa.x && pa.y >= pa.z ) n.y = copysignf( 1.f , p.y ) ;              
                else if( pa.z >= pa.x && pa.z >= pa.y ) n.z = copysignf( 1.f , p.z ) ;              
                */

                float3 n = make_float3(1.f,0.f,0.f);  


                shading_normal = geometric_normal = n ;
                instanceIdentity = identity ;
                rtReportIntersection(0);
            } 
        }
    }
}







//void intersect_box(quad& q0, quad& q1, quad& q2, quad& q3, const uint4& identity)

static __device__
void intersect_box(quad& q0, const uint4& identity)
{
    // using q0 to derive bbox
    const float3 min_ = make_float3(q0.f.x - q0.f.w, q0.f.y - q0.f.w, q0.f.z - q0.f.w ); 
    const float3 max_ = make_float3(q0.f.x + q0.f.w, q0.f.y + q0.f.w, q0.f.z + q0.f.w ); 
    const float3 cen_ = make_float3(q0.f.x, q0.f.y, q0.f.z) ;    

    float3 t0 = (min_ - ray.origin)/ray.direction;
    float3 t1 = (max_ - ray.origin)/ray.direction;

    // slab method 
    float3 near = fminf(t0, t1);
    float3 far = fmaxf(t0, t1);
    float tmin = fmaxf( near );
    float tmax = fminf( far );

    if(tmin <= tmax && tmax > 0.f) 
    {
        bool check_second = true;
        float tint = tmin > 0.f ? tmin : tmax ; 

        if(rtPotentialIntersection(tint))
        {
            float3 p = ray.origin + tint*ray.direction - cen_ ; 
            float3 pa = make_float3(fabs(p.x), fabs(p.y), fabs(p.z)) ;
            //float pmax = fmaxf(pa);

            float3 n = make_float3(0.f) ;
            if(      pa.x >= pa.y && pa.x >= pa.z ) n.x = copysignf( 1.f , p.x ) ;              
            else if( pa.y >= pa.x && pa.y >= pa.z ) n.y = copysignf( 1.f , p.y ) ;              
            else if( pa.z >= pa.x && pa.z >= pa.y ) n.z = copysignf( 1.f , p.z ) ;              


            shading_normal = geometric_normal = n ;
            instanceIdentity = identity ;
            if(rtReportIntersection(0)) check_second = false ;   // material index 0 
        } 

        // handle when inside box, or are epsilon near clipped 
        if(check_second)
        {
            if(rtPotentialIntersection(tmax))
            {
                float3 p = ray.origin + tmax*ray.direction - cen_ ; 
                float3 pa = make_float3(fabs(p.x), fabs(p.y), fabs(p.z)) ;
                //float pmax = fmaxf(pa);

                float3 n = make_float3(0.f);  

                if(      pa.x >= pa.y && pa.x >= pa.z ) n.x = copysignf( 1.f , p.x ) ;              
                else if( pa.y >= pa.x && pa.y >= pa.z ) n.y = copysignf( 1.f , p.y ) ;              
                else if( pa.z >= pa.x && pa.z >= pa.y ) n.z = copysignf( 1.f , p.z ) ;              

                shading_normal = geometric_normal = n ;
                instanceIdentity = identity ;
                rtReportIntersection(0);
            } 
        }
    }
}





