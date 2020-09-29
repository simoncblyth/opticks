geometry_model_review
=======================

Objective of this review
--------------------------

High level description of the GGeo geometry model:
how it is constructed, issues with mm0.

See Also 
----------

* :doc:`identity_review.rst`
* :doc:`GGeo-OGeo-identity-direct-review.rst` Mechanics of the direct workflow:

Relevant Tests
----------------

GGeoConvertTest 
    does GGeo::dryrun_convert checking what OGeo::convert will do 
GGeoIdentityTest
    access identity across all mm 
GGeoTest 


Summary of issue with Geometry Model
--------------------------------------

The root cause of the problems are trying to do too much in GMergedMesh slot 0 (aka mm0).

It tries to carry both:

1. all volume "global" information
2. non-instanced "remainder" information

It kinda gets away with this conflation by splitting on high-level/low-level axis using 
"selected" volumes.
But the result is still confusing even when it can be made to work, so it is 
prone to breakage.


globalinstance just adding to confusion
-------------------------------------------

In Aug I added an extra mm slot, called the GlobalInstance, which 
treats the remainder geometry just like instanced. That was motivated 
by identity problems.

TODO: change the GlobalInstance -> RemainderInstance  

* BUT: This step might just be adding confusion. 
* MUST: make GMergedMesh simpler by doing less in it 

::

     415 void GMergedMesh::countVolume( GVolume* volume, bool selected, unsigned verbosity )
     416 {
     417     const GMesh* mesh = volume->getMesh();
     418 
     419     // with globalinstance selection is honoured at volume level too 
     420     bool admit = ( m_globalinstance && selected ) || !m_globalinstance ;  
     421     if(admit)
     422     {
     423         m_num_volumes += 1 ; 
     424     }
     425     if(selected)
     426     {
     427         m_num_volumes_selected += 1 ;
     428         countMesh( mesh ); 
     429     }   
     430     
     431     //  hmm having both admit and selected is confusing 
     432     


Idea for solution : keep the "all" info in GNodeLib arrays and get it persisted there 
---------------------------------------------------------------------------------------

Need to have access to both "all" geometry info and "remainder" geometry info.
Both these currently in GMergedMesh slot zero. 

Investigate the usage of the "all" info from mm0, 

1. relocating all volume info collection into GNodeLib::

    void GNodeLib::add(const GVolume* volume)

2. get it persisted there (currently just persists pvlist lvlist names) for all volumes a


This would help by moving GMergedMesh in the simpler direction.


STEPS TO MINIMIZE DUPLICATION
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. simplify GMergedMesh using self contained code pull offs into other classes, eg GVolume


BUT : do not want to duplicate code in GMergedMesh and GNodeLib
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* move GVolume->array content processing to GNodeLib : use NPY arrays with glm::vec help
* GMergedMesh then takes it from there with the appropriate selections  
* where appropriate do things in static methods of other classes





High Level Geometry Information Flow
----------------------------------------

0. Geant4 volume tree
1. recursive traversal of Geant4 tree (eg by X4PhysicalVolume::convertNode) yields GGeo::m_root tree of GVolume(GNode)
2. GInstancer labels the tree of GVolume with ridx (repeat index) integers with zero being for the non-instanced remainder
3. GMergedMesh::Create for each slot collects volumes for each instance and the remainder volumes into separate MM.






