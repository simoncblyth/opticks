LHCb_Rich_Lucas_unclear_sphere_phisegment_issue
==================================================

Commits
---------


* https://bitbucket.org/simoncblyth/opticks/commits/96579e4c6

* https://bitbucket.org/simoncblyth/opticks/commits/466cee37f

::

    epsilon:opticks blyth$ git commit -m "generalize the handling of planar center-extent-gensteps allowing to slice other axes, use XY slice to look at G4Sphere phi segment with xxs.sh "
    [master 96579e4c6] generalize the handling of planar center-extent-gensteps allowing to slice other axes, use XY slice to look at G4Sphere phi segment with xxs.sh
     9 files changed, 501 insertions(+), 65 deletions(-)


    epsilon:opticks blyth$ git commit -m "add X4Solid_intersectWithPhiSegment_debug_mode for debug variations in the intersectWithPhiSegment geometry "
    [master 466cee37f] add X4Solid_intersectWithPhiSegment_debug_mode for debug variations in the intersectWithPhiSegment geometry
     13 files changed, 472 insertions(+), 279 deletions(-)
     create mode 100755 CSGOptiX/cxr_pub.sh
    epsilon:opticks blyth$ git push 


Lucas phicut thetacut
-------------------

* https://bitbucket.org/imjavert/opticks/src/master/CSG/csg_intersect_node.h



Idea for implementing G4Sphere theta and phi segments using unbounded primitives CSG_PHICUT CSG_THETACUT
----------------------------------------------------------------------------------------------------------


Hi Lucas, 

I took at look at your code with:

    https://bitbucket.org/simoncblyth/opticks/src/master/extg4/tests/convertSphereTest.sh

    https://bitbucket.org/simoncblyth/opticks/src/master/extg4/tests/convertSphereTest.hh

    https://bitbucket.org/simoncblyth/opticks/src/master/extg4/tests/convertSphereTest.cc



epsilon:tests blyth$ ./convertSphereTest.sh
...

NTreeAnalyse height 3 count 9
              in                    

      di              in            

  sp      sp      zs          in    

                          zs      zs



I note that your theta segment has the wrong idea of the shape of G4Sphere theta segments.
Lines of constant theta result in cones as opposed to lines of constant phi which 
result in vertical plane chops.

If you install pyvista you can visualize in 3D the Geant4 intersects onto solids using 

    cd ~/opticks/extg4
    ./xxs.sh 

For example using theta_start 0.25 theta_delta 0.5  (units of pi)
creates a sphere with both sides of a z-axial cone through zero removed.

https://simoncblyth.bitbucket.io/env/presentation/xxs/G4Sphere_theta_segment_0_half.png


Your theta mask based on zsphere has no chance of matching Geant4.
You will need to create cone masks to do that.

However I think the approach you are taking is a very expensive 
way to do things, especially if the segmented sphere had to be used 
in CSG combination with other solids. 

For fast ray tracing you want the CSG trees to be a simple as possible.
So how to make things simpler...

For one, you are using zsphere hemispheres just for their endcaps to act as chopping planes. 
It is simpler and cheaper just to use planes. 
But planes by themselves are rather low level to work with. 

So, actually I think the way to do this is start by implementing 
two new primitives:


    CSG_PHICUT
          phi_start
          phi_delta

          Simply two rotatable half-planes attached “vertically” to the z-axis



    CSG_THETACUT
          theta_start
          theta_delta  

          theta_start = 0,         theta_delta < 0.5 pi     1 cone on +z   
          theta_start = 0, 0.5 pi < theta_delta < pi        complemented cone on -z
       
          additional cones might be needed or a symmetrical cone is a possibility 



The starting point for implementing those is to understand existing 
primitive implementations in opticks/CSG/csg_intersect_node.h

  

 298 INTERSECT_FUNC
 299 bool intersect_node_cone( float4& isect, const quad& q0, const float t_min , const float3& ray_origin, const float3& ray_direction )
 300 {
 301     float r1 = q0.f.x ;
 302     float z1 = q0.f.y ;
 303     float r2 = q0.f.z ;
 304     float z2 = q0.f.w ;   // z2 > z1
 305 
 306     float tth = (r2-r1)/(z2-z1) ;
 307     float tth2 = tth*tth ;
 308     float z0 = (z2*r1-z1*r2)/(r1-r2) ;  // apex
 309 
 310 #ifdef DEBUG
 311     printf("//intersect_node.h:cone r1 %10.4f z1 %10.4f r2 %10.4f z2 %10.4f : z0 %10.4f \n", r1, z1, r2, z2, z0 );
 312 #endif
 313 
 314     float r1r1 = r1*r1 ;
 315     float r2r2 = r2*r2 ;
 316 
 317     const float3& o = ray_origin ;
 318     const float3& d = ray_direction ;
 319 
 320     //  cone with apex at [0,0,z0]  and   r1/(z1-z0) = tanth  for any r1,z1 on the cone
 321     //
 322     //     x^2 + y^2  - (z - z0)^2 tanth^2 = 0 
 323     //     x^2 + y^2  - (z^2 -2z0 z - z0^2) tanth^2 = 0 
 324     //
 325     //   Gradient:    [2x, 2y, (-2z tanth^2) + 2z0 tanth^2 ] 
 326     //
 327     //   (o.x+ t d.x)^2 + (o.y + t d.y)^2 - (o.z - z0 + t d.z)^2 tth2 = 0 
 328     // 
 329     // quadratic in t :    c2 t^2 + 2 c1 t + c0 = 0 
 330 
 331     float c2 = d.x*d.x + d.y*d.y - d.z*d.z*tth2 ;
 332     float c1 = o.x*d.x + o.y*d.y - (o.z-z0)*d.z*tth2 ;
 333     float c0 = o.x*o.x + o.y*o.y - (o.z-z0)*(o.z-z0)*tth2 ;
 334     float disc = c1*c1 - c0*c2 ;
 335 
 336 #ifdef DEBUG
 337     printf("//intersect_node.h:cone c2 %10.4f c1 %10.4f c0 %10.4f disc %10.4f : tth %10.4f \n", c2, c1, c0, disc, tth  );
 338 #endif
 339 
 340 

...


If you want to try to implement an “intersect_node_thetacut”, 
I suggest you start by restricting yourself to theta_start = 0. 
as then there is just one cone to worry about so it will be very similar to the above “intersect_node_cone”, 
the only difference being that its simpler because an infinite cone is needed, so there is no endcap to implement
and the parameters will need to be changed to theta_start, delta_theta.

Does your LHCb RICH geometry actually need the theta cut ? 

Implementing “intersect_node_phicut” will be simpler as it should be 
similar to “intersect_node_plane” and “intersect_node_slab”.


Simon



Added Debug modes to X4Solid::intersectWithPhiSegment
--------------------------------------------------------

::

     867     int debug_mode = intersectWithPhiSegment_debug_mode ;
     868     LOG(error)
     869         << " startPhi " << startPhi
     870         << " deltaPhi " << deltaPhi
     871         << " segZ " << segZ
     872         << " segR " << segR
     873         << " debug_mode " << debug_mode
     874         ;
     875
     876     if( debug_mode == 1 )
     877     {
     878         LOG(error) << "X4Solid_intersectWithPhiSegment_debug_mode " << debug_mode << " RETURNING SEGMENT " ;
     879         result = segment ;
     880         result->label = BStr::concat(m_name, "_debug_segment", NULL);
     881     }
     882     else if( debug_mode == 2 )
     883     {
     884         LOG(error) << "X4Solid_intersectWithPhiSegment_debug_mode " << debug_mode << " RETURNING UNION " ;
     885         result = nnode::make_operator(CSG_UNION, whole, segment);
     886         result->label = BStr::concat(m_name, "_debug_union", NULL);
     887     }
     888     else if( debug_mode == 3 )
     889     {
     890         LOG(error) << "X4Solid_intersectWithPhiSegment_debug_mode " << debug_mode << " RETURNING DIFFERENCE " ;
     891         result = nnode::make_operator(CSG_DIFFERENCE, whole, segment);
     892         result->label = BStr::concat(m_name, "_debug_difference", NULL);
     893     }
     894 
     895 
     896     return result ;
     897 }


GeoChain/run.sh::

     85 if [ "${GEOM/SphereWithPhiSegment}" != "$GEOM" ] ; then
     86 
     87 
     88    export X4Solid_convertSphere_enable_phi_segment=1
     89 
     90    return_segment=1
     91    return_union=2
     92    return_difference=3
     93    export X4Solid_intersectWithPhiSegment_debug_mode=$return_difference
     94 
     95    env | grep X4Solid
     96 fi


::

    gc ; ./run.sh 
    cx ; ./cxr_geochain.sh   # need to edit for pick up _Darwin geometries
    

    EYE=1,1,1 TMIN=0.1 ./cxr_geochain.sh        

    ## adjust viewpoint to see the segment in difference mode 
    ## it looks like the z-extent of the wedge is at least half what it should be 


::

    2021-12-05 15:45:32.397 ERROR [3803680] [*X4Solid::intersectWithPhiSegment@868]  startPhi 0 deltaPhi 90 segZ 101 segR 150 debug_mode 3
    2021-12-05 15:45:32.397 ERROR [3803680] [*X4Solid::intersectWithPhiSegment@890] X4Solid_intersectWithPhiSegment_debug_mode 3 RETURNING DIFFERENCE 

::

     525 
     526     float segZ = radius*1.01 ;
     527     float segR = radius*1.5 ;
     528 
     529     nnode* result =  has_deltaPhi && enable_phi_segment
     530                   ?
     531                      intersectWithPhiSegment(ret, startPhi, deltaPhi, segZ, segR )
     532                   :
     533                      ret
     534                   ;
     535 






CSG_GGeo_Convert::convertNode failing for lack of CSGNode::setAABBLocal with convexpolyhedron
------------------------------------------------------------------------------------------------

* FIXED : by special casing convexpolyhedron bbox setup as unlike other prim cannot easily get bbox just from the param 

* unlike other primitives it is not so easy to get the bbox from just the planes
  of the nconvexpolyhedron (would have to reconstruct vertices by intersecting all the planes with each other to do that) 

* but nconvexpolyhedron::make_segment sets the bbox into the nconvexpolyhedron 
  object at creation as the starting point is essentially the vertices 

  * TODO: use this in the cg conversion establishing communication from NConvexPolyhedron to CSGNode 
    without CSGNode having the NConvexPolyhedron header
  

::

    2021-12-04 16:06:09.265 INFO  [2922416] [GParts::getTypeMask@1507]  primIdx 0 partOffset 0 numParts 3
     partIdx    0 tc    2 tm          4 tag   in
     partIdx    1 tc    5 tm         32 tag   sp
     partIdx    2 tc   19 tm     524288 tag   co
    2021-12-04 16:06:09.265 INFO  [2922416] [*CSG_GGeo_Convert::convertPrim@335]  primIdx    0 meshIdx    0 comp.getTypeMask 4 CSG::TypeMask in  CSG::IsPositiveMask 1
    2021-12-04 16:06:09.265 FATAL [2922416] [CSGNode::setAABBLocal@363]  not implemented for tc 19 CSG::Name(tc) convexpolyhedron
    Assertion failed: (0), function setAABBLocal, file /Users/blyth/opticks/CSG/CSGNode.cc, line 364.

    Process 58616 launched: '/usr/local/opticks/lib/GeoChainSolidTest' (x86_64)
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff69ccab66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff69e95080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff69c261ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff69bee1ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x0000000101cc415e libCSG.dylib`CSGNode::setAABBLocal(this=0x000000010b000080) at CSGNode.cc:364
        frame #5: 0x000000010070463a libCSG_GGeo.dylib`CSG_GGeo_Convert::convertNode(this=0x00007ffeefbfda60, comp=0x000000010871be60, primIdx=0, partIdxRel=2) at CSG_GGeo_Convert.cc:477
        frame #6: 0x0000000100704f7b libCSG_GGeo.dylib`CSG_GGeo_Convert::convertPrim(this=0x00007ffeefbfda60, comp=0x000000010871be60, primIdx=0) at CSG_GGeo_Convert.cc:372
        frame #7: 0x00000001007058f6 libCSG_GGeo.dylib`CSG_GGeo_Convert::convertSolid(this=0x00007ffeefbfda60, repeatIdx=0) at CSG_GGeo_Convert.cc:264
        frame #8: 0x0000000100706069 libCSG_GGeo.dylib`CSG_GGeo_Convert::convertAllSolid(this=0x00007ffeefbfda60) at CSG_GGeo_Convert.cc:133
        frame #9: 0x0000000100703ef0 libCSG_GGeo.dylib`CSG_GGeo_Convert::convertGeometry(this=0x00007ffeefbfda60, repeatIdx=-1, primIdx=-1, partIdxRel=-1) at CSG_GGeo_Convert.cc:120
        frame #10: 0x0000000100703835 libCSG_GGeo.dylib`CSG_GGeo_Convert::convert(this=0x00007ffeefbfda60) at CSG_GGeo_Convert.cc:75
        frame #11: 0x00000001000ddc87 libGeoChain.dylib`GeoChain::convertSolid(this=0x00007ffeefbfe010, so=0x0000000108500400, meta_="creator:GeoChainSolidTest\nname:SphereWithPhiSegment\ninfo:WITH_PMTSIM \n") at GeoChain.cc:70
        frame #12: 0x000000010000e85b GeoChainSolidTest`main(argc=3, argv=0x00007ffeefbfe718) at GeoChainSolidTest.cc:84
        frame #13: 0x00007fff69b7a015 libdyld.dylib`start + 1
        frame #14: 0x00007fff69b7a015 libdyld.dylib`start + 1
    (lldb) 




Issue
------

Hi Lucas, 

> Hello! I am a student working with a team based at RAL/CERN to upgrade the
> simulation of LHCb's RICH detector to make use of Opticks. I am contacting you
> to request some information on a section of Opticks and potentially offer a fix
> to a bug that looks to exist in the current build (provided my idea for a fix
> works once I have the correct syntax).  

Fixes are very welcome.

> Unfortunately we are currently having
> some trouble with conversions in the geometry between Geant4 and Opticks,
> something I have narrowed down to a problem with the sphere class used in the
> conversion.  Notably, Opticks has functionality to trim solids by an angle in
> phi disabled, and when re-enabled manually it causes the sphere generated to
> simply disappear. Having the phi angle untrimmed is a bug that in most projects
> would be largely unnoticeable, but due to certain specifications of the
> detector we are simulating in our case it is completely debilitating.

Phi-segmenting has not been important for the geometries I have used
Opticks with so far, so this feature has not been well tested and thus
bugs are highly likely, and doubly so because the feature is disabled.


> After testing a few fixes to this myself, I have found that the present method
> of implementing phi angles is not working because primitives in Opticks don't
> seem to interact well with custom polyhedra from the NConvexPolyhedra class.  

You need to be more specific. 

There is not need to "interact", the primitives are put together within CSG intersection 
so all that needs to happen is for the phi segment shape to provide ray trace intersects
and be in the correct position to cut into the other primitive.
The most probable bugs are for the Opticks shape not to match the Geant4 one due to 
the Opticks idea of what the phi segment parameters mean not matching the Geant4 idea. 

I welcome your assistance to debug the phi segmenting feature.
You can help by creating a test that demonstrates the issue. 

> I have extensively checked this against both the currently used prism function
> and a custom made sphere segment function, neither of which worked. I would
> suggest this may be caused by these custom solids being translated incorrectly
> when generated, as they will intersect occasionally if the custom solid is
> correctly oriented AND larger than the entire detector geometry - but I
> digress, the specifics are not important, it suffices to say these do not work
> as intended at present.

Such specifics are vital to identifying the cause of the bug. 
 
> Fortunately I have found a possible way of fixing this using primitives
> instead, which I initially considered just for spheres but have since realised
> may be applicable for any of the other shapes with this issue (there seem to be
> several in X4Solid.cc which have the same problem). This would involve
> generating a base sphere using the same theta-angle-only NZSphere class as
> before, then rotating it to align with the axis in phi and performing an
> intersection with the initial sphere (or other solid). 

You are suggesting to change the phi segment implementation 
from using NConvexPolyhedron(set of planes) to using an appropriately 
rotated z-cut sphere ?

An interesting idea, but my intuition suggests that could only model a 
very small subset of possible phi-segments.  
 
> Having tested this I can
> confirm that the two solids will intersect immediately without issue when
> generated (unlike with NConvexPolyhedra), so in theory this should work.

You need to support your words with working code in order to convince me.
 
> The important section of this email begins here: The only issue at present is
> that I do not know how to rotate the transform of a solid in Opticks, so cannot
> complete the fix without this.  My supervisor does not seem keen on the idea
> that this may be the problem and the other members of the team do not have the
> experience in Opticks to know how to do this, so I have decided to contact you
> directly in hopes you may be able to help.


Opticks NNode trees can have transforms assigned to any node. 
This is just an index within the NNode that points to the transform
that gets added to another array.
So to add a transform you will have to multiply the rotation transform
by any other transform (in the correct order) that is already associated 
to the node.

However I am unconvinced that this is the way to go.  
NConvexPolyhedon does work on its own (eg with ray traced trapezoids, tetrahedrons and icosahedrons) 
so it can be made to work in CSG combination, it just takes some effort to get the phi segment "cheese" shape 
to be in the right position for the phi segmenting to match Geant4. 
  

> To give you an idea of the rotation I need to perform, here is a comment I
> found within the file NNode.cpp which may have been written by you: "To
> translate or rotate a surface modeled as an SDF, you can apply the inverse
> transformation to the point before evaluating the SDF." I can understand why
> that would work, but I do not know on what transform I could enact the rotation
> on to do this. 

SDF usage within NNode is just for debugging.  However an equivalent thing 
is done by the ray tracing implementation where to ray trace a transformed
primitive you first apply an inverse transform to all the rays. 

> The function and class I have been editing is
> X4Solid::convertSphere_() in extg4/X4Solid.cc, where the line used to generate
> the solid is equivalent to cn = make_zsphere( x, y, z, radius, zmin, zmax ) ;
> 
> If you happen to know the transform in this class/function I would need to
> perform the rotation on, perhaps as well as the command to perform such a
> rotation, your help would be incredibly beneficial. 

I remain of the opinion that a z-cut sphere can only provide
a very small subset of possible phi segments. 


> I hope you are doing well and thank you for taking out the time to read this.
> Any help or advice would be greatly appreciated.
> 
> Best regards, Lucas Girardey
> 
> P.S. My apologies if this email was rather long and overwritten, I am told I do
> that quite often and only hope this wasn't much of an imposition.

Learning to communicate succintly and convincingly takes experience.
The trick for doing this is to provide or refer to runnable code, 
as that is the most definite way to communicate.

A picture may be worth a thousand words, but runnable code is worth a million pictures. 

Simon



x4/xxs.sh X4GeometryMaker::make_SphereWithPhiSegment
-------------------------------------------------------

Added "SphereWithPhiSegment" to extg4/xxs.sh to see exactly what Geant4 
means by the phi segment params.

* TODO: single genstep emanating 3D rays and 3D pyvista presentation of intersects
* TODO: apply the GeoChain to SphereWithPhiSegment and look for issues with the translation + ray trace intersects
* TODO: actually after reviewing G4Sphere a 2D x-y look will show the phi-segment clearly  


G4Sphere : ePhi end-phi cPhi center-phi 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

    inline
    void G4Sphere::InitializePhiTrigonometry()
    {
      hDPhi = 0.5*fDPhi;                       // half delta phi
      cPhi  = fSPhi + hDPhi;
      ePhi  = fSPhi + fDPhi;

      sinCPhi    = std::sin(cPhi);
      cosCPhi    = std::cos(cPhi);
      cosHDPhiIT = std::cos(hDPhi - 0.5*kAngTolerance); // inner/outer tol half dphi
      cosHDPhiOT = std::cos(hDPhi + 0.5*kAngTolerance);
      sinSPhi = std::sin(fSPhi);
      cosSPhi = std::cos(fSPhi);
      sinEPhi = std::sin(ePhi);
      cosEPhi = std::cos(ePhi);
    }






Hi Lucas

I do not think your idea is workable (comments on that below), 
however thank you for communicating about the issue as it motivated 
me to look into X4Solid::convertSphere and allowed me to fix a problem 
with the phi segmenting, and to realise a discrepancy between Opticks and
Geant4 in theta segmenting that is unresolved.

In order to debug phi segments I switched from intersection 
with the segment to difference with it. The two below renders are before 
and after fixing the z-extent of the segment wedge. 
The segment was half the size it needed to be in z.

https://simoncblyth.bitbucket.io/env/presentation/CSGOptiXRender/GeoChain_Darwin/SphereWithPhiSegment/cvd0/50001/cxr_geochain/cam_1/cxr_geochain_SphereWithPhiSegment_difference_old.jpg
https://simoncblyth.bitbucket.io/env/presentation/CSGOptiXRender/GeoChain_Darwin/SphereWithPhiSegment/cvd0/50001/cxr_geochain/cam_1/cxr_geochain_SphereWithPhiSegment_difference_new.jpg


> I understand this would be helpful to show how I know the current system is
> malfunctioning, but unfortunately I don't think I'd be able to do this in a
> format that I could send for you to trial yourself. My current mode of testing
> has been by performing modifications to the geometry of our simulated detector
> and recording the location of registered hits, giving myself a very rudimentary
> way of visualising changes in geometry under different parameters and changes
> to the sphere function. It's not ideal I know, but with enough trials it has
> revealed a lot of small details which would otherwise have been missed.

To make progress in development it is vital to learn to create small focussed 
test executables (effectively "unit tests") that exercise one feature/issue. 
This allows you to communicate with precision using executable code, rather than 
with vast swathes of text, that most potential readers will not have the 
patience to read in depth.

Also critically it gives you a fast development cycle for investigations.

> That is what I had initially thought also, which is why I ignored it at first
> and tried to see if I could fix the polyhedral implementation. However, just by
> chance I showed my supervisor the inputs for the NZSphere class and he said
> something I hadn't originally considered.  

NZSphere does "flat" z-cuts, restricting the z-range of the sphere.
Thats very different shape to G4Sphere thetaStar thetaDelta.  
This difference is the unresolved discrepancy between Opticks and Geant4
wrt theta segments.

> The sphere only takes inputs between
> the angles of 0 and 180 degrees, 

G4Sphere theta is 0->180,  phi is 0->360

> and therefore it technically has no way of
> knowing (besides convention) which side of the sphere you want the solid to be
> generated on. He suggested that since it doesn't know that, there's every
> chance that the theta segments generated are mirrored down the sphere's centre.
> I assumed this couldn't be the case, but when I went on to check in our own
> geometry, sure enough it was there. 

I do not follow this argument, to explain you will need to draw some diagrams, 
or make some renders.   

G4Sphere phiStart phiDelta results in "vertical" chops parallel to the z-axis   

> Apologies for not being able to give code as proof, but I feel with a little
> explanation the rather crude image attached to this email should suffice to
> explain at least a little of how I have found this.  In this setup there's only
> two surfaces being interacted with, both of which are totally efficient in
> Opticks, so all photons impacting will be absorbed. Both surfaces are spheres
> segmented in theta but not phi, one large (top left), one small (bottom right),
> and the source of photons is a positron emitting via the Cerenkov process. The
> particle passes through both surfaces before the stepping process halts.
> Ignoring the large sphere to begin with, there is a small collection of hits
> just above the expected distribution of hits on the small sphere. In the
> default version of this geometry the two spheres are mirrors, and having
> checked repeatedly it is definitely the case that the lower side of the sphere
> is correctly oriented (if required I can provide evidence that I have
> absolutely confirmed this is correct for the case being tested). That small dot
> of points just above the expected surface on the lower sphere absolutely should
> not be there, and is a result of this mirroring effect.  The reason we don't
> see it from the other sphere is because its radius is large enough that its
> opposite side is outside of the extent of the detector, and therefore never
> intersects the path - something which is normally true of the opposing side of
> the smaller sphere when it is at its correct scale.

Trying to debug something in such a contorted way is not practical.
You need a simple situation and simple code path in order to have any chance of 
identifying causes of bugs. 

> I digress, the point of mentioning this was to explain why this would enable my
> original idea to work.  Using the aforementioned make_zsphere( x, y, z, radius,
> zmin, zmax ) function, if we automatically set maxz = radius, this corresponds
> to an angle of Phi = 0, thus giving no separation between the two mirrored
> halves. From here, we could set minz = radius * cos( 0.5 * deltaPhi ), halving
> the angle of the mirror generated on each side. 

I think you have wrong idea about the shape of NZSphere. 
But thank you for raising this : as it made me realise that 
the Opticks theta segmenting does not match Geant4.

NZSphere simply takes a sphere and chops it in z. 
It is not an appropriate shape for making phi segments, other than
making a hemispeher and using it as a chopping plane : which is what 
the nconvexpolyhedron segment is doing anyhow.

>  No angle between them with two
> mirrored halves of angle 0 to 180 gives us an easy way of making a whole
> sphere, which we can then account for the rotation of afterwards. In Geant4 the
> phi angle of spheres is defined between +-180 degrees, thus putting the zero
> point in the same place between the two geometries. All we'd have to do to
> account for the difference now is add to the angle of startPhi to correct for
> the fact that under normal circumstances the angle has to account for the
> centring of the mirror (that is, we change startPhi += 0.5 * deltaPhi).
> Perform one rotation to align phi with the correct axis, then another to rotate
> the now correctly aligned phi to set startPhi correctly (although I expect in
> many cases this angle will be zero anyway).  
>
> Finally, take the intersection
> with the existing segment in theta and boom, we now have a working spherical
> segmentation that matches between Geant4 and Opticks - one which may also be
> able to replace the current segmenting function that doesn't appear to be
> working (and which would already be rather limited, effectively only working as
> far as 90 degrees).

Intersecting with a wedge shape to phi-segment is not restricted to 90 degrees
because you can just increase the "radius" extent of the segment. 
But I do see this as problematic as it needs to be very large in order to get to 180 
and going beyond 180 would not work.
A solution for that problem would be to implement an unbounded CSG_SEGMENT shape 
which comprises just two planes that intersect on the z-axis.   

Using unbounded shapes works fine so long as they are always intersected with 
or subtracted. There are already implementations of unbounded primitives
CSG_PLANE and CSG_SLAB (two parallel planes).


> It would also save a lot more time performing trial and error on the polyhedron
> class to work out where it's going wrong; I'm sure it'd be useful to know, but
> if it can be avoided I'd say its worth a try.
>
> I understand this all sounds very Optimistic and that the result I've shown as
> evidence of this having the possibility of working looks like the computer
> generated equivalent of a drawing in crayons, but so long as there is no
> problem caused during the rotation, this should work.  
>
> Again, I cannot support them with any finished code since I'm not sure on
> performing a rotation which is the current problem; I can however offer you
> what I have so far. This can at least show that the code may be used to
> generate a full sphere, and that the two spheres generated (for theta and phi
> respectively) are generated from the same point, thus meaning they already
> intersect without having to be moved (can be observed by generating one as a
> full sphere and the other with a lower angle). Here is my version of the
> convertSphere_() function, most lines are identical to your own but I figured
> I'd send the whole thing incase I missed something.

I think you idea is a non-starter as it is based on a mis-understanding 
of the Opticks NZSphere shape. 

To convince me otherwise you will need to make a better argument with 
diagrams and preferably with renders of geometry. 

Of course this raises the question of how to implement in Opticks an 
equivalent for the G4Sphere theta segment functionality. The way it 
is done currently with NZSphere is wrong.   
An immediate idea is to subtract cones from the sphere. 

If you need that functionality feel free to try to implement it. 
I may be able to incorporate your work into Opticks.

Simon

For notes on my investigations of the issues you pointed out see

https://bitbucket.org/simoncblyth/opticks/src/master/notes/issues/LHCb_Rich_Lucas_unclear_sphere_phisegment_issue.rst

