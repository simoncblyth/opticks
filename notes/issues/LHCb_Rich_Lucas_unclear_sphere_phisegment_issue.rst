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




 
