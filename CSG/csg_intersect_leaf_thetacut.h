#pragma once

/**
SCB comments on intersect_leaf_thetacut_lucas

1. normalize(isect) a float4 is a bug : you are requiring isect.w to be zero 

2. you say same maths as intersect_node_cone (now intersect_leaf_cone)
   but you use entirely different language 

3. invalidate candidates by setting to t_min is needed for the shape to 
   work in CSG combinations as need expected behaviour as t_min is varied



intersect_leaf_thetacut_lucas
--------------------------------
Based on same maths behind intersect_node_cone, see there for explanation.

WORKS FOR 0 <= THETA <= 180 BUT BEWARE: USER NEEDS TO BE CAREFUL WHEN DEFINING QUAD, MUST BE SET
    //    q.f.x = theta0 == 0.5 ? 0.0 : cos(theta0 * pi ) / abs(cos(theta0 * pi));
    //    q.f.y = theta0 == 0.5 ? 0.0 : tan(theta0 * pi) * tan(theta0 * pi);
    //    q.f.z = theta1 == 0.5 ? 0.0 :  cos(theta1 * pi) / abs(cos(theta1 * pi));
    //    q.f.w = theta1 == 0.5 ? 0.0 : tan(theta1 * pi) * tan(theta1 * pi);
    // if .x and .z are not set 0.0 cos(...) float inaccuracy will mean plane not recognised.
    // if .y and .w are not set 0.0 magnitudes will give wacky values, not worth the risk.
    
**/
LEAF_FUNC
bool intersect_leaf_thetacut_lucas(float4& isect, const quad& thetaDat, const float t_min, const float3& rayOrigin, const float3& rayDirection)
{   //thetaData contains x = cos(theta0)/abs(cos(theta0)), y = tan^2 (theta0), z = cos(theta1)/abs(cos(theta1)), w = tan^2 (theta1)

    float dirMag = rayDirection.x * rayDirection.x + rayDirection.y * rayDirection.y - rayDirection.z * rayDirection.z * thetaDat.f.y;
    float originDirMag = rayOrigin.x * rayDirection.x + rayOrigin.y * rayDirection.y - rayOrigin.z * rayDirection.z * thetaDat.f.y;
    float originMag = rayOrigin.x * rayOrigin.x + rayOrigin.y * rayOrigin.y - rayOrigin.z * rayOrigin.z * thetaDat.f.y;
    float disc = originDirMag * originDirMag - originMag * dirMag;

    bool intersects = disc > 0.f; 
    float discRoot = intersects ? sqrt(disc) : 0.f; //avoids sqrt(NEGATIVE)

    float t_cand = intersects ? (-originDirMag + discRoot) / dirMag : RT_DEFAULT_MAX; //beginning on t_cand saves defining extra variable

    if (thetaDat.f.x * (t_cand * rayDirection.z + rayOrigin.z) < 0.f || t_cand <= t_min) t_cand = RT_DEFAULT_MAX; //eliminates bad t_cand/mirror cone 

    float t0 = intersects ? (-originDirMag - discRoot) / dirMag : RT_DEFAULT_MAX;
    if (thetaDat.f.x * (t0 * rayDirection.z + rayOrigin.z) > 0.f && t0 > t_min) t_cand = fminf(t_cand, t0); 
    //works here since t_cand will already be either valid or INF

    dirMag += rayDirection.z * rayDirection.z * (thetaDat.f.y - thetaDat.f.w);
    originDirMag += rayOrigin.z * rayDirection.z * (thetaDat.f.y - thetaDat.f.w);
    originMag += rayOrigin.z * rayOrigin.z * (thetaDat.f.y - thetaDat.f.w);
    disc = originDirMag * originDirMag - originMag * dirMag;

    intersects = disc > 0.f;
    discRoot = intersects ? sqrt(disc) : 0.f;

    t0 = intersects ? (-originDirMag + discRoot) / dirMag : RT_DEFAULT_MAX;
    if (thetaDat.f.z * (t0 * rayDirection.z + rayOrigin.z) > 0.f && t0 > t_min) t_cand = fminf(t_cand, t0);

    const float t1 = intersects ? (-originDirMag - discRoot) / dirMag : RT_DEFAULT_MAX;
    if (thetaDat.f.z * (t1 * rayDirection.z + rayOrigin.z) > 0.f && t1 > t_min) t_cand = fminf(t_cand, t1);


    const float t_plane = -rayOrigin.z / rayDirection.z;
    const bool plane = thetaDat.f.x * thetaDat.f.z == 0.0 && t_plane > t_min && t_cand > t_plane;
    const bool valid = t_cand < RT_DEFAULT_MAX || plane;

    if (valid) {
        const bool side = t_cand == t0 || t_cand == t1; //corrects normals for both cones/planes around 90 degrees

        isect.x = plane ? 0.0 : (side ? thetaDat.f.z * (rayOrigin.x + t_cand * rayDirection.x)
                                       : - thetaDat.f.x * (rayOrigin.x + t_cand * rayDirection.x));
        isect.y = plane ? 0.0 : (side ? thetaDat.f.z * (rayOrigin.y + t_cand * rayDirection.y)
                                       : - thetaDat.f.x * (rayOrigin.y + t_cand * rayDirection.y));
        isect.z = plane ? (thetaDat.f.x == 0.0 ? 1.0 : -1.0)
                        : ( side ? - thetaDat.f.z * (rayOrigin.z + t_cand * rayDirection.z) * thetaDat.f.w
                                 : thetaDat.f.x * (rayOrigin.z + t_cand * rayDirection.z) * thetaDat.f.y );
        isect = normalize(isect);
        isect.w = plane ? t_plane : t_cand;
    }

    return valid;
}

