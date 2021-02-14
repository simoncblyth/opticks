optix7_migration
===================

Tasks
-------

1. GAS formation needs bbox bounds CPU side



GGeo -> OptiX 7 : mapping between geometry models 
-----------------------------------------------------

Aiming for simplest possible OptiX 7 geometry of form:: 

    IAS
       tr0_00 -> GAS_0     ## (N0=1) only one identity transform from remainder volumes GAS


       tr1_00 -> GAS_1     ## (N1=10->30k) transforms for others but simple GAS, 5~10 bi
       tr1_01 -> GAS_1
       tr1_02 -> GAS_1
       ...
       tr1_N1 -> GAS_1


       tr2_00 -> GAS_2
       tr2_01 -> GAS_2
       tr2_02 -> GAS_2
       ...
       tr2_N2 -> GAS_2


       tr3_00 -> GAS_3     ## expect only ~10 different GAS for JUNO geometry 
       tr3_01 -> GAS_3
       tr3_02 -> GAS_3
       ...
       tr3_N3  -> GAS_3


Where::

       GAS_0
          bi0:bi1:bi2:...:bi~300
          (1) (1) (1)    :(1)         <-- SBT entries 
          (hmm GAS_0 the remainder volumes will be big, it could be split into multiple)

       GAS_1
          bi0:bi1:bi2:
          (1) (1) (1)

       GAS_2
          bi0:bi1
          (1) (1)

       GAS_3 
       ..
       GAS_9


GAS
   1:1 with GMergedMesh
   OptixBuildInput 1:1 with single GParts from one solid (just a local bbox)

IAS
   gets instance transforms from all GMergedMesh, what about GNodeLib 
   for the all info : does it have the geometry reference ? 

SBT
   number of HitGroup SBT entries is same as the the number of distinct solids  


Is a ``GParts::treeBBox`` based on the persisted arrays needed, or can NCSG do that already ?  
--------------------------------------------------------------------------------------------------

* OptiX 7 GAS inputs need the bbox CPU side unlike OptiX 6 which uses ``csg_intersect_part.h::csg_bounds_prim`` 

* ``nnode::bbox`` which traverses the ``nnode`` tree 

  * ``nnode::get_composite_bbox`` 
  * ``nnode::get_primitive_bbox``

As this needs the ``nnode`` tree, it may be pre-cache only.


What about triangulated ?
---------------------------

Start with analytic as that is more critical, follow 
exactly the same structure with triangulated just changing 
the OptixBuildInput from the custom bbox to the triangulated vertices buffers etc..

Hmm with triangulated there is more performance pressure to merge. 
But triangulated is so fast anyway.

Mixed analytic and triangulated ?
-------------------------------------

* there is a case for a mostly analytic geometry with a few triangulated standalone torii (JUNO guide tube)
* presumably just make a separate GAS for those


Is a GParts::UnCombine needed ?
------------------------------------

Reviewing the analytic geometry GPts/GParts etc
there is no need for a ``GParts::UnCombine`` since are now 
deferring ``GParts::Create`` to ``GGeo::deferredCreateGParts``
can directly use the original single sold ``GParts`` simply by 
collecting them in ``GParts::m_subs``.  This is because 
deferring GParts creation means the subs vector does not need 
to survive the cache.

