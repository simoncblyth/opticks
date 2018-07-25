extend-opticks-nnt-to-direct-translation
==========================================

motivation
------------

*opticks-nnt* code generation producing python/CSG and nnode/C++ for 
each solid and the bash tbool for running them was extremely useful 
for debugging booleans in the old route.  So want to extend that one 
step backwards and generate Geant4 geometry code from the nnode model : 
to allow automated clean room testing of each individual G4 solid used 
in a geometry and its translation to the GPU.  Would want to test in 
both unbalanced and balanced forms : as balancing is a probable
cause of issues.

how to implement ?
~~~~~~~~~~~~~~~~~~~~~~

Each nnode primitive subclass+operator type will need 
methods analogous to the *analytic/csg.py* `content_*` 
returning strings and something like the below in the base 
to bring them all together::

    void nnode::as_g4code(std::ostream& out) const 

Follow pattern of GMeshLib::reportMeshUsage for stream mechanics.

I would prefer to not have this code in nnode for clarity but that 
seems difficult initially. So start in nnode by look for opportunites to 
to factor it off.  This can yield a string of g4code for each node
of the tree. 

Note there is no G4 dependency : are just generating strings that 
happen to be Geant4 geometry code.


alt approach : from "System 1" following lunch 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Doing this in X4Solid will be much easier than nnode:

* have the real Geant4 instance right there to spill its beans
* this code gen is just for testing : so no problem to do it only at conversion 
  and the resulting string can be stored along with the nnode

* but this doesnt help with testing of tree balancing ? BUT its so much 
  faster to implement than the nnode approach that its worth doing if first, checking 
  all solids from this perspective and then worrying about tree balancing later.

::

    implemented via X4SolidBase::setG4Param and X4SolidBase::g4code which gets invoked
    by the setG4Param


what remains to implement
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* transforms, displaced solids



how to test ?
~~~~~~~~~~~~~~

Start in x4:tests/X4SolidTest then move to okg4:tests/OKX4Test 
once nearly done.

OKX4Test
    messy multi-Opticks boot using CGDMLDetector, then converts
    with X4PhysicalVolume populating a 2nd GGeo instance.


how to persist ?
~~~~~~~~~~~~~~~~~~

Start with adhoc writers on nnode/NCSG that take a path, may need 
to go all way up to GParts for "official" writes into geocache.

 

X4PhysicalVolume
-------------------

The recursive *X4PhysicalVolume::convertNode* invokes *X4Solid::Convert* on first meeting each solid::

    506      if(mh->csgnode == NULL)
    507      {
    508          // convert G4VSolid into nnode tree and balance it if overheight 
    509 
    510          nnode* tree = X4Solid::Convert(solid)  ;
    511          nnode* result = NTreeProcess<nnode>::Process(tree, nd->soIdx, lvIdx);
    512 
    513          mh->csgnode = result ;



review opticks-nnt codegen
-------------------------------

During the old python gdml2gltf conversion the geometry codegen is 
invoked as part of saving the csg into the gltf extras.

*opticks.analytic.csg:as_code* invokes node recursive *as_code_r* that 
converts the tree parsed from the GDML into python/CSG or C++/nnode geometry description::

    1335     def as_tbool(self, name="esr"):
    1336         tbf = TBoolBashFunction(name=name, root=self.alabel, body=self.as_code(lang="py")  )
    1337         return str(tbf)
    1338 
    1339     def as_NNodeTest(self, name="esr"):
    1340         nnt  = NNodeTestCPP(name=name, root=self.alabel, body=self.as_code(lang="cpp")  )
    1341         return str(nnt)

*opticks.analytic.csg:save* writes the generated code to file::

 793         self.write_tbool(lvidx, tboolpath)
 794 
 795         nntpath = self.nntpath(treedir, lvidx)
 796         self.write_NNodeTest(lvidx, nntpath)
 797 
 798         nodepath = self.nodepath(treedir)
 799         np.save(nodepath, nodebuf)


These are written inside the extras of the old glTF::

    [blyth@localhost ~]$ opticks-tbool-info

    opticks-tbool-info
    ======================

       opticks-tbool-path 0 : /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras/0/tbool0.bash
       opticks-nnt-path 0   : /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras/0/NNodeTest_0.cc





