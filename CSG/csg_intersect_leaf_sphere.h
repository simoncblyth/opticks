#pragma once

LEAF_FUNC
float distance_leaf_sphere(const float3& pos, const quad& q0 )
{
    float3 center = make_float3(q0.f);
    float radius = q0.f.w;
    float3 p = pos - center;
    float sd = length(p) - radius ; 
    return sd ; 
}


LEAF_FUNC
void intersect_leaf_sphere( bool& valid_isect, float4& isect, const quad& q0, const float& t_min, const float3& ray_origin, const float3& ray_direction )
{
    float3 center = make_float3(q0.f);
    float radius = q0.f.w;

    float3 O = ray_origin - center;
    float3 D = ray_direction;

    float b = dot(O, D);
    float c = dot(O, O)-radius*radius;
    float d = dot(D, D);

#ifdef CATASTROPHIC_SUBTRACTION_ROOTS
    float disc = b*b-d*c;   // when d*c small,  sdisc ~ b => catastrophic precision loss in root2 = (-b + sdisc)/d
    float sdisc = disc > 0.f ? sqrtf(disc) : 0.f ;   // repeated root for sdisc 0.f
    float root1 = (-b - sdisc)/d ;
    float root2 = (-b + sdisc)/d ;  // root2 > root1 always
#else
    float root1, root2, disc, sdisc ;   
    robust_quadratic_roots(root1, root2, disc, sdisc, d, b, c ) ; //  Solving:  d t^2 + 2 b t +  c = 0    root2 > root1 
#endif

    float t_cand = sdisc > 0.f ? ( root1 > t_min ? root1 : root2 ) : t_min ;

    valid_isect = t_cand > t_min ;
    if(valid_isect)
    {
        isect.x = (O.x + t_cand*D.x)/radius ;   // normalized by construction
        isect.y = (O.y + t_cand*D.y)/radius ;
        isect.z = (O.z + t_cand*D.z)/radius ;
        isect.w = t_cand ;
    }

#ifdef DEBUG
    //printf("//intersect_leaf_sphere valid_isect %d  isect ( %10.4f %10.4f %10.4f %10.4f)  \n", valid_isect, isect.x, isect.y, isect.z, isect.w ); 
    printf("//intersect_leaf_sphere valid %d radius %10.4f center (%10.4f, %10.4f, %10.4f) ray_ori (%10.4f, %10.4f, %10.4f)  \n", 
       valid_isect,  radius, center.x, center.y, center.z, ray_origin.x, ray_origin.y, ray_origin.z  );  
#endif


}


