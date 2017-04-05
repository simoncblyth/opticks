CSG Transform Support
=========================


Transforms with Ray-trace and SDF
------------------------------------

To translate or rotate a surface modeled as an CSG tree, 
apply the inverse transformation to the point for SDFs or the ray for 
raytracing before doing the SDF distance calc or ray tracing intersection
calc.


SDF
------

* Where to hold the transform in nnode trees and CSG trees ?

 * G4 allows the RHS of a boolean combination to be transformed using 
   a transform that lives with the combination



* use glm::mat4 ?



