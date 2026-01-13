#pragma once

/**
intersect_leaf_halfspace
--------------------------

Define CSG_HALFSPACE with a unit normal *n* and a distance *w* from the origin
in the normal direction. Hence points *p* that are within the plane fulfil::

   p.n = w

   \  *
   +\  *
   ++\  *
   +++\  *
   ++++\  *
   +++++\  *
   ++++++\  *               n     outwards pointing normal, away from halfspace
   +++++++\  *           .
   ++++++++\  *       .
   +++++++++\  *   .
   ++++++++++\  .
   +++++++++++O  *
   ++++++++++++\  *
   +++++++++++++\  *
   ++ inside ++++\  *     outside
   +++++++++++++++\  *
   ++++++++++++++++\  *             +->
   +++++++++++++++++\  *            d.n > 0 : direction with normal : heading away from halfspace
   ++++++++++++++++++\  *           o.n > w : origin outside halfspace
   +++++++++++++++++++\  *          [no isect, dn*(on-w) > 0]
   ++++++++++++++++++++\  *
   +++++++++++++++++++++\  *        <-+
   ++++++++++++++++++++++\  *       d.n < 0 : direction against normal : heading towards halfspace
   +++++++++++++++++++++++\  *      o.n > w : origin outside halfspace
   ++++++++++++++++++++++++\  *     [poss isect, dn*(on-w) < 0]
                            \  *
   +->                       \  *
   d.n > 0                    \  *
   o.n < w                     \  *
   [poss isect, dn*(on-w) < 0]  \  *
                                 \  *           +
   <-+                            \  *           \   d.n = 0     : direction parallel to plane, perpendicular to normal
   d.n < 0                         \  *           \  o.n - w > 0 : origin outside halfspace
   o.n < w                          \  *             [no isect, no exit at infinity as not inside]
   [no isect, dn*(on-w) > 0]         \  *
   [exit at infinity]                 \  *
                                       \  +
                                        \  \      d.n = 0          : d perp to n
       +                                 \  \     o.n - w = 0      : o within plane
        \                                 \  *    [infinite isects, exit at infinity]
         \                                 \  *
        d.n = 0     : d perp to n
        o.n - w < 0 : o inside
        [no isect, exit at infinity]


Parametric ray::

     p = o + d t



Intersects of ray with plane::

     p.n = w

     (o + d t).n = w


                   ( w - o.n )
     =>       t =  -------------
                      d.n


Rays with origin inside the halfspace that are headed
away from the plane cannot intersect with it.
But the CSG intersection algorithm needs to:

1. classify every ray constituent trial into ENTER/EXIT/MISS
2. every shape needs an "other" side

This these rays can be regarded to EXIT the halfspace at infinity.
This is an UNBOUNDED_EXIT.

      o.n < w  ,  (on-w) < 0   origin inside halfspace
      d.n < 0                  d into halfspace [no isect, exit at infinity]
      d.n = 0                  d parallel to plane [no isect, exit at infinity]

      o.n = w  ,  (on-w) = 0   origin on plane
      d.n < 0                  d into halfspace [invalid t=0 isect, exit at infinity]
      d.n = 0                  d within plane, perp to normal [infinite isect, exit at infinity]

Can select both the above situations with "<="::

     ( on-w ) <= 0 && dn <= 0



Consider a test halfspace, defined by n [1/sqrt(2), 1/sqrt(2), 0, 0]
The normal points away from the halfspace, so intersecting a cylinder
with axis in z-direction with this expect to get half a cylinder::

           Y
   + .     |      n
   + + .   |    .
   + + + . | .
   + + + + 0------X
   + + + + +.
   + + + + + +.
   + + + + + + +.
   + + + + + + + +.

Set GEOM to "LocalPolyconeWithPhiCut" and do the conversion::

    ~/o/g4cx/tests/G4CX_U4TreeCreateCSGFoundryTest.sh

Viewing that from different positions::

    NOXGEOM=1 EYE=4,0,10 UP=0,1,0 cxr_min.sh                                   # expected half cylinder, with cut side pointing to +x+

    NOXGEOM=1 EYE=0,0,10 EXTENT_FUDGE=2 CAM=orthographic UP=0,1,0 cxr_min.sh   # UNEXPECTED FULL CYLINDER
    NOXGEOM=1 EYE=1e-50,0,10 EXTENT_FUDGE=2 CAM=orthographic UP=0,1,0 cxr_min.sh  # UNEXPECTED FULL CYLINDER
    NOXGEOM=1 EYE=1e-40,0,10 EXTENT_FUDGE=2 CAM=orthographic UP=0,1,0 cxr_min.sh  # UNEXPECTED FULL CYLINDER
    NOXGEOM=1 EYE=1e-37,0,10 EXTENT_FUDGE=2 CAM=orthographic UP=0,1,0 cxr_min.sh  # UNEXPECTED FULL CYLINDER

    NOXGEOM=1 EYE=1e-36,0,10 EXTENT_FUDGE=2 CAM=orthographic UP=0,1,0 cxr_min.sh  # expected half cylinder
    NOXGEOM=1 EYE=1e-35,0,10 EXTENT_FUDGE=2 CAM=orthographic UP=0,1,0 cxr_min.sh  # expected half cylinder
    NOXGEOM=1 EYE=1e-30,0,10 EXTENT_FUDGE=2 CAM=orthographic UP=0,1,0 cxr_min.sh  # expected half cylinder
    NOXGEOM=1 EYE=0.001,0,10 EXTENT_FUDGE=2 CAM=orthographic UP=0,1,0 cxr_min.sh # expected half cylinder
    NOXGEOM=1 EYE=0.01,0,10 EXTENT_FUDGE=2 CAM=orthographic UP=0,1,0 cxr_min.sh # expected half cylinder
    NOXGEOM=1 EYE=0.1,0,10 EXTENT_FUDGE=2 CAM=orthographic UP=0,1,0 cxr_min.sh # expected half cylinder
    NOXGEOM=1 EYE=0.2,0,10 EXTENT_FUDGE=2 CAM=orthographic UP=0,1,0 cxr_min.sh # expected half cylinder
    NOXGEOM=1 EYE=0.5,0,10 EXTENT_FUDGE=2 CAM=orthographic UP=0,1,0 cxr_min.sh # expected half cylinder
    NOXGEOM=1 EYE=1,0,10 EXTENT_FUDGE=2 CAM=orthographic UP=0,1,0 cxr_min.sh   # expected half cylinder
    NOXGEOM=1 EYE=4,0,10 EXTENT_FUDGE=2 CAM=orthographic UP=0,1,0 cxr_min.sh   # expected half cylinder
    NOXGEOM=1 EYE=20,0,10 EXTENT_FUDGE=2 CAM=orthographic UP=0,1,0 cxr_min.sh  # still expected


After using eps 1e-7 gets worse::

    NOXGEOM=1 EYE=1e-6,0,10 EXTENT_FUDGE=2 CAM=orthographic UP=0,1,0 cxr_min.sh   # UNEXPECTED FULL CYLINDER
    NOXGEOM=1 EYE=1e-5,0,10 EXTENT_FUDGE=2 CAM=orthographic UP=0,1,0 cxr_min.sh   # expected half cylinder

Tweaking, not much diff::

    NOXGEOM=1 EYE=1e-8,0,10 EXTENT_FUDGE=2 CAM=orthographic UP=0,1,0 cxr_min.sh   # UNEXPECTED FULL CYLINDER
    NOXGEOM=1 EYE=1,0,10 EXTENT_FUDGE=2 CAM=orthographic UP=0,1,0 cxr_min.sh   # expected half cylinder

Proceed to CSG/tests/csg_intersect_prim_test.sh which so far has not reproduced the issue.

**/


LEAF_FUNC
void intersect_leaf_halfspace( bool& valid_isect, float4& isect, const quad& q0, const float t_min, const float3& o, const float3& d )
{
    float3 n = make_float3(q0.f.x, q0.f.y, q0.f.z);
    float w = q0.f.w;

    float on = dot(o,n);
    float dn = dot(d,n);
    float on_w = on - w;
    float adn = fabsf(dn);

    float eps = 1e-9f;
    bool inside = on_w < -eps ;

    float t = adn > 0.f ? -on_w / dn : t_min ;
    valid_isect = t > t_min ;

    if (valid_isect)
    {
        isect.x = n.x ;
        isect.y = n.y ;
        isect.z = n.z ;
        isect.w = t ;
    }
    else
    {
        if (inside)
        {
            isect.y = -0.f;
        }
    }

#ifdef DEBUG_HALFSPACE
    bool yflip = valid_isect == false && isect.y == -0.f ;
    printf("//intersect_leaf_halfspace n [%8.3f,%8.3f,%8.3f,%8.3f] o [%8.3f,%8.3f,%8.3f] d[%8.3f,%8.3f,%8.3f] on/dn/on_w/adn [%8.3f,%8.3f,%8.3f,%8.3f] t %8.3f valid_isect %d inside %d yflip %d \n",
          n.x, n.y, n.z, w, o.x, o.y, o.z, d.x, d.y, d.z, on,dn,on_w,adn,eps,t, valid_isect, inside, yflip);
#endif

}

