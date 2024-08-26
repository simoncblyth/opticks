flexible_forced_triangulation
================================

Issue
------

Previously geometry code used simple layout::

    ridx:0  global analytic compound solid 
    ridx:1,2,3,...  instanced analytic solids

To provide flexible triangulation need:

1. generalize ana/tri type of each ridx slot  
2. user control to make an lv become triangulated 

Relevant context
-------------------

::

    sysrap/stree.h 
    u4/U4Tree.h 



Review progress
----------------

What configures force triangulation ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    stree__force_triangulate_solid


::

    epsilon:u4 blyth$ opticks-f stree__force_triangulate_solid
    ./sysrap/stree.h:    static constexpr const char* stree__force_triangulate_solid = "stree__force_triangulate_solid" ; 
    ./sysrap/stree.h:    force_triangulate_solid(ssys::getenvvar(stree__force_triangulate_solid,nullptr)), 
    ./sysrap/stree.h:Uses the optional comma delimited stree__force_triangulate_solid envvar list of unique solid names
    ./sysrap/stree.h:depending on the "stree__force_triangulate_solid" envvar list of unique solid names. 
    epsilon:opticks blyth$ 


TODO: test this with a script 



Where are nds/rem/tri collected ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

U4Tree::initNodes_r does initial collection from Geant4 into *nds*, 
subsequently the *rem* and *tri* subsets are populated by stree::collectGlobalNodes
which is invoked at the tail of stree::factorize


stree::get_ridx_type
~~~~~~~~~~~~~~~~~~~~~~~

::

    git diff ed7ced230^-1

     
         int      get_num_ridx() const ;  
    +    int      get_num_remainder() const ; 
    +    int      get_num_triangulated() const ;
    +    char     get_ridx_type(int ridx) const ;
 


where is stree::get_ridx_type used to effect the force triangulation ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First need to import the stree to form the CSGFoundry geom, made changes::

    CSGImport::importSolid
    CSGImport::importSolidGlobal
    CSGImport::importSolidFactor
    
Then need to convert from CSGFoundry geom into GAS/SBT.

* HMM: how to detect triangulated from the solid ? 
* Nope not possible directly, unless use the label eg: r0 f1 f2 f3 t4



how did the old CSGFoundry level trimesh post hoc switch to tri ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


With CSGFoundry::isSolidTrimesh::

     314 #ifdef WITH_SOPTIX_ACCEL
     315 void SBT::createGAS(unsigned gas_idx)
     316 {
     317     SOPTIX_BuildInput* bi = nullptr ;
     318     SOPTIX_Accel* gas = nullptr ;
     319 
     320     bool trimesh = foundry->isSolidTrimesh(gas_idx);  // post-hoc triangulation 
     321     const std::string& label = foundry->getSolidLabel(gas_idx);
     322 


HMM: can/should I co-opt the old CSGFoundry::isSolidTrimesh to adopt force triangulation ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* looks like it 



where are the stree::rem used ? 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



TODO: generalize old layout assuming code ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


eg::

     82 void CSGImport::importSolid()
     83 {
     84     int num_ridx = st->get_num_ridx() ;
     85     for(int ridx=0 ; ridx < num_ridx ; ridx++)
     86     {
     87         std::string _rlabel = CSGSolid::MakeLabel('r',ridx) ;
     88         const char* rlabel = _rlabel.c_str();
     89 
     90         if( ridx == 0 )
     91         {
     92             importSolidRemainder(ridx, rlabel );
     93         }
     94         else
     95         {
     96             importSolidFactor(ridx, rlabel );
     97         }
     98     }
     99 }





U4Tree.h
----------

U4Tree::initSolids_Mesh 
    All solids have analytic and triangulated forms. The tri/ana fork happens later.  


CSGFoundry::isSolidTrimesh HUH : TOO LATE TO DO THIS HERE ?
------------------------------------------------------------

Yep, its too late to do this within CSG. 
This was for primitive post hoc trimesh control. 

Earlier control used in stree::collectGlobalNodes

* NB simplifying assumption that all configured tri nodes are global (not instanced)


