alternative_geometry_definition_using_a_signed_distance_function_SDF
=====================================================================


Using SDFs to define your geometry has the advantage of being novel
- it should yield some interesting conference presentations.
But the novelty has the disadvantage that you will have to implement
everything yourself and you will struggle to get benefits from
industry optimizations like NVIDIA OptiX acceleration structures.

For finding intersection with SDF look for "Sphere Tracing" by John Hart.
But note that its iterative - you only get close to the actual answer.
Plus sphere tracing has troubles with rays close to parallel with your surfaces
as the algorithm then takes loads of small steps.
Some interesting SDF and sphere tracing links:

* https://iquilezles.org/articles/raymarchingdf/
* https://graphics.stanford.edu/courses/cs348b-20-spring-content/uploads/hart.pdf
* https://ianthehenry.com/posts/periodic-spaces/
* https://iquilezles.org/articles/distfunctions/

Note that even some very simple shapes like the ellipsoid do not have a simple SDF.
Space warping approximations cause problems. Any space warping is problematic -
as the closest distance in the warped space will not generally be closest
in unwarped.

SDFs (especially periodic ones) have the disadvantages of inflexibility
that would likely make it impractical to work with a real geometry or
to optimize various choices of fibre arrangements - other than just changing
pitch of a regular X-Y-Z grid.

Implementing many complex CSG shapes using just min and max combinations of
distance functions - looks very appealing, but that leads to discontinuities
at seams which can cause you intersection to have trouble converging.
Search for "Rvachev Functions" or R-functions to learn more on a mathematical
approach that may avoid some of the problems of simple min/max combinations.

You might think that warping space would enable you to handle non-regular
arrangements - but that will likely violate the Lipschitz condition which
has to be fullfilled for SDF to yield valid intersects.  Violations
can also cause loss of efficiency.

Another consideration is that the SDF just gives you a distance - you have
no opportunity to provide identity info. For example Opticks ray trace
intersects onto analytic or triangulated geometry provide a boundary index
from which material and surface indices can be obtained. With SDF you would
need separate lookup to give identity info from the coordinate rather than
directly having that.


