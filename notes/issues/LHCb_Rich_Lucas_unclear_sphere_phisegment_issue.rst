LHCb_Rich_Lucas_unclear_sphere_phisegment_issue
==================================================


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






Hi Simon,

Thanks for your fast response. I'll try to answer the concerns you've raised as
succinctly as possible.

There is not need to "interact", the primitives are put together within CSG
intersection so all that needs to happen is for the phi segment shape to
provide ray trace intersects and be in the correct position to cut into the
other primitive.  The most probable bugs are for the Opticks shape not to match
the Geant4 one due to the Opticks idea of what the phi segment parameters mean
not matching the Geant4 idea.

Apologies, when I said interact I should have said intersect, I think I used
that word because I was considering the problem in terms of what I have suspect
may be happening - I.E. that the polyhedron may be being translated
incorrectly, due to something like a mismatch between G4 and OK like you said. 

Such specifics are vital to identifying the cause of the bug.

It's a little tangential to say but it's refreshing to hear you affirm this,
others on the team I'm on have been treating it as though the smaller details
do not matter; as such I didn't want to bore you with the details initially,
but if required I could explain what I did to arrive at my suggestions. 

You can help by creating a test that demonstrates the issue.

I understand this would be helpful to show how I know the current system is
malfunctioning, but unfortunately I don't think I'd be able to do this in a
format that I could send for you to trial yourself. My current mode of testing
has been by performing modifications to the geometry of our simulated detector
and recording the location of registered hits, giving myself a very rudimentary
way of visualising changes in geometry under different parameters and changes
to the sphere function. It's not ideal I know, but with enough trials it has
revealed a lot of small details which would otherwise have been missed.

An interesting idea, but my intuition suggests that could only model a very
small subset of possible phi-segments.

That is what I had initially thought also, which is why I ignored it at first
and tried to see if I could fix the polyhedral implementation. However, just by
chance I showed my supervisor the inputs for the NZSphere class and he said
something I hadn't originally considered.  The sphere only takes inputs between
the angles of 0 and 180 degrees, and therefore it technically has no way of
knowing (besides convention) which side of the sphere you want the solid to be
generated on. He suggested that since it doesn't know that, there's every
chance that the theta segments generated are mirrored down the sphere's centre.
I assumed this couldn't be the case, but when I went on to check in our own
geometry, sure enough it was there. 

Apologies for not being able to give code as proof, but I feel with a little
explanation the rather crude image attached to this email should suffice to
explain at least a little of how I have found this.  In this setup there's only
two surfaces being interacted with, both of which are totally efficient in
Opticks, so all photons impacting will be absorbed. Both surfaces are spheres
segmented in theta but not phi, one large (top left), one small (bottom right),
and the source of photons is a positron emitting via the Cerenkov process. The
particle passes through both surfaces before the stepping process halts.
Ignoring the large sphere to begin with, there is a small collection of hits
just above the expected distribution of hits on the small sphere. In the
default version of this geometry the two spheres are mirrors, and having
checked repeatedly it is definitely the case that the lower side of the sphere
is correctly oriented (if required I can provide evidence that I have
absolutely confirmed this is correct for the case being tested). That small dot
of points just above the expected surface on the lower sphere absolutely should
not be there, and is a result of this mirroring effect.  The reason we don't
see it from the other sphere is because its radius is large enough that its
opposite side is outside of the extent of the detector, and therefore never
intersects the path - something which is normally true of the opposing side of
the smaller sphere when it is at its correct scale.

I digress, the point of mentioning this was to explain why this would enable my
original idea to work.  Using the aforementioned make_zsphere( x, y, z, radius,
zmin, zmax ) function, if we automatically set maxz = radius, this corresponds
to an angle of Phi = 0, thus giving no separation between the two mirrored
halves. From here, we could set minz = radius * cos( 0.5 * deltaPhi ), halving
the angle of the mirror generated on each side. No angle between them with two
mirrored halves of angle 0 to 180 gives us an easy way of making a whole
sphere, which we can then account for the rotation of afterwards. In Geant4 the
phi angle of spheres is defined between +-180 degrees, thus putting the zero
point in the same place between the two geometries. All we'd have to do to
account for the difference now is add to the angle of startPhi to correct for
the fact that under normal circumstances the angle has to account for the
centring of the mirror (that is, we change startPhi += 0.5 * deltaPhi).
Perform one rotation to align phi with the correct axis, then another to rotate
the now correctly aligned phi to set startPhi correctly (although I expect in
many cases this angle will be zero anyway).  Finally, take the intersection
with the existing segment in theta and boom, we now have a working spherical
segmentation that matches between Geant4 and Opticks - one which may also be
able to replace the current segmenting function that doesn't appear to be
working (and which would already be rather limited, effectively only working as
far as 90 degrees). 

It would also save a lot more time performing trial and error on the polyhedron
class to work out where it's going wrong; I'm sure it'd be useful to know, but
if it can be avoided I'd say its worth a try.

I understand this all sounds very Optimistic and that the result I've shown as
evidence of this having the possibility of working looks like the computer
generated equivalent of a drawing in crayons, but so long as there is no
problem caused during the rotation, this should work.  

You need to support your words with working code in order to convince me.

Again, I cannot support them with any finished code since I'm not sure on
performing a rotation which is the current problem; I can however offer you
what I have so far. This can at least show that the code may be used to
generate a full sphere, and that the two spheres generated (for theta and phi
respectively) are generated from the same point, thus meaning they already
intersect without having to be moved (can be observed by generating one as a
full sphere and the other with a lower angle). Here is my version of the
convertSphere_() function, most lines are identical to your own but I figured
I'd send the whole thing incase I missed something.

::

    nnode* X4Solid::convertSphere_(bool only_inner)
    {
       const G4Sphere* const solid = static_cast<const G4Sphere*>(m_solid);

       float rmin = solid->GetInnerRadius()/mm ; 
       float rmax = solid->GetOuterRadius()/mm ; 

       bool has_inner = !only_inner && rmin > 0.f ; 
       nnode* inner = has_inner ? convertSphere_(true) : NULL ;  
       float radius = only_inner ? rmin : rmax ;   

       LOG(verbose) 
                 << " radius : " << radius 
                 << " only_inner : " << only_inner
                 << " has_inner : " << has_inner 
                 ;

       float startThetaAngle = solid->GetStartThetaAngle()/degree ; 
       float deltaThetaAngle = solid->GetDeltaThetaAngle()/degree ; 

       // z to the right, theta   0 -> z=r, theta 180 -> z=-r
       float rTheta = startThetaAngle ;
       float lTheta = startThetaAngle + deltaThetaAngle ;
       assert( rTheta >= 0.f && rTheta <= 180.f) ; 
       assert( lTheta >= 0.f && lTheta <= 180.f) ; 

       bool zslice = startThetaAngle > 0.f || deltaThetaAngle < 180.f ; 

       LOG(verbose) 
                 << " rTheta : " << rTheta
                 << " lTheta : " << lTheta
                 << " zslice : " << zslice
                 ;

       float x = 0.f ; 
       float y = 0.f ; 
       float z = 0.f ; 

       nnode* cn = NULL ; 
       if(zslice)
       {
           double zmin = radius*std::cos(lTheta*CLHEP::pi/180.) ;
           double zmax = radius*std::cos(rTheta*CLHEP::pi/180.) ;
           assert( zmax > zmin ) ; 
           cn = make_zsphere( x, y, z, radius, zmin, zmax ) ;
           cn->label = BStr::concat(m_name, "_nzsphere", NULL) ; 
       }
       else
       {
           cn = make_sphere( x, y, z, radius );
           cn->label = BStr::concat(m_name, "_nsphere", NULL ) ; 
       }

       nnode* ret = has_inner ? nnode::make_operator(CSG_DIFFERENCE, cn, inner) : cn ; 
       if(has_inner) ret->label = BStr::concat(m_name, "_ndifference", NULL ) ; 


       float startPhi = solid->GetStartPhiAngle()/degree ; 
       float deltaPhi = solid->GetDeltaPhiAngle()/degree ; 
       bool has_deltaPhi = deltaPhi < 360.f ; 



       nnode* result = NULL;

       if(has_deltaPhi)
         {
    //if has phi

           double zminPhi = radius*std::cos(0.5 * deltaPhi * CLHEP::pi/180.) ;
           double zmaxPhi = radius;
        /*
    sets maximum and minimum z in cylindrical coordinates. here we exploit the mirrored generation of the cylindrical coords with zero angle between the two sections of half length to create a single sliced wheel of the correct size at their intersection. from here we may rotate this new wedge into the correct position.
         */
        
          double startPhiAdjust = startPhi + 0.5 * deltaPhi; //adjusts for centre

           assert( zmaxPhi > zminPhi ) ; //checks for deltaPhi<0 
        
           //Rotation of root here maybe?

        nnode* segmentPhi = NULL ; //create nnode

           segmentPhi = make_zsphere( x, y, z, 1.01 * radius, zminPhi, zmaxPhi ) ;
        //generates segment

           segmentPhi->label = BStr::concat(m_name, "_nzsphere", NULL) ; 
        //labels segment

           //Counter rotation of root here, or rotation of segment if root not rotated. 

        result = nnode::make_operator(CSG_INTERSECTION, ret, segmentPhi);
       } else {

         result = ret;

       }

       return result ; 
    }

I hope that this may at least convince you there is a possibility this would
work. I await your response, but will continue looking for a solution to this
in the meantime. I hope you are doing well, and thank you again for your
correspondence.

Many thanks,
Lucas 
