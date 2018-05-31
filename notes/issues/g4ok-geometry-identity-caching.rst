g4ok-geometry-identity-caching
================================

Objectives
-----------

1. verify geocache selected by the user (name/directory argument/envvar) 
   matches the live G4 geometry as a sanity check 

2. provide a digest name for new geometry  
    

Thoughts on geometry identity
------------------------------

* "full" digest identity for a geometry is beyond the scope (would need to 
   visit every member variable of every instance of the geometry), and 
   recursively do the same for pointers : and there would be special 
   cases so would need to be implemented within each class 

   * unless can form a digest directly by viewing the member variables
     of an instance as a sequence of bits with pointers excluded 
     and followed recursively.  Thats only feasible via introspection.  
   
* BUT partial digest is still useful, eg combining 
  sub-digests for all materials, surfaces, structure transforms, etc..


Approaches to geometry identity : digest tree? 
--------------------------------------------------

Brute force approach to geometry identity is to export to GDML (without pointer addresses) 
and compute a hash of the file.  That is excessively slow for large geometries, also not 
very helpful at explaining what changed.

Great potential for short-circuited "early-exit" comparisons (eg just object counts) 
if instead of forming a single digest from lots of others form a tree of digests that identify 
eg each material, surface, solid, volume.  Each material has multiple 
properties any of which might have been changed so have to decide at what level of
the tree to descend to. 


How the geometry identity fits in to Opticks workflows
--------------------------------------------------------

Workflows differ in how much to automate and how much 
to hand over control to the user, eg if 
rely on users knowledge of when geometry is changed : can 
get away with rudimentary sanity check only.


* first Opticks run on a geometry, creates geocache in a 
  directory with name including a digest that represents the geometry identity 

* either:

  1. user needs to provide the name of that directory in a subsequent run
  in order to reuse the cache


* rely on user to identify the cache (using a digest including dirpath) that 
  matches the 


