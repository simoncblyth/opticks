Geometry Translation
======================

Outline of the steps to translate Geant4 geometry to OptiX GPU Geometry

1. analyse geometry to find different types of repeated instances of geometry together
   with the global geometry of solids that are not repeated enough to pass the 
   instancing cuts.  Repeats that are contained within other repeats are disqualified 
   in order to end up with "assemblies" of multiple volumes. 
   This for example finds the ~5 volumes that comprise the JUNO PMTs and 
   all their 4x4 transforms. 

2. convert each G4VSolid into a Opticks nnode/NCSG tree 

3. balance the NCSG tree by: 

   a) positivization : removing all subtractions in the tree by application of DeMorgans 
      rules pushing negations into complemented primitives makes the tree easier to
      rearrange as it then contains commutative unions or intersections only

   b) rearrange the tree to make more balanced

   Balancing the tree is needed as many boolean solids (eg repeated subtractions) 
   yield imbalanced trees which are inefficiently handled as complete binary trees. 

4. serialize the CSG tree of each solid into buffers

5. serialize the structure of the full geometry into buffers for each instance
   as well as for the global non-instanced geometry

6. interleave all material and surface properties as a function of wavelength 
   into a buffer ready for conversion into a GPU texture   

7. apply the NVIDIA OptiX API to put the entire geometry into GPU buffers


