# === func-gen- : graphics/openmeshtest/openmeshtest fgp graphics/openmeshtest/openmeshtest.bash fgn openmeshtest fgh graphics/openmeshtest
openmeshtest-src(){      echo graphics/openmeshtest/openmeshtest.bash ; }
openmeshtest-source(){   echo ${BASH_SOURCE:-$(env-home)/$(openmeshtest-src)} ; }
openmeshtest-vi(){       vi $(openmeshtest-source) ; }
openmeshtest-env(){      elocal- ; }
openmeshtest-usage(){ cat << EOU

OpenMeshTest 
=============

Developing mesh surgery code.

Done
-----

#. convert NPY meshes into OpenMesh 
#. partition G4 created triangle soup (did Assimp do any diddling?) 
   into real connected V - E + F = 2  Euler Polyhedra 

TODO
-----

#. geometrical comparison of two component meshes to identify close faces, 
   
   * for each face of mesh A find parallel faces using 
     absolute normal dot products in mesh B
   * compare distances in normal direction between candidate aligned faces 
   * also check standard distance between barycenters of candidates
   * devise criteria to pick the cleaved faces of mesh A and B

#. delete the cleaved faces from A and B, make sure mesh boundary is 
   navigable

#. find way to weld together A and B by joining up the boundary
   need distance criterior to decide whether to fuse or put in 
   edges to do the welding  

   * de-nudge in z ?

   * suspect the volume in question will need new edges going
     outwards in (x,y) almost fixed in z : as 
     the cleave happened along a z flange plane 
     (polycone with 2 identical z planes)

#. convert surgeried back into NPY format 

#. rearrange code into lib for actual usage

#. test in ggv-



EOU
}
openmeshtest-dir(){  echo $(env-home)/graphics/openmeshtest ; }
openmeshtest-idir(){ echo $(local-base)/env/graphics/openmeshtest ; }
openmeshtest-bdir(){ echo $(openmeshtest-idir).build ; }

openmeshtest-cd(){   cd $(openmeshtest-dir); }
openmeshtest-icd(){  cd $(openmeshtest-idir); }
openmeshtest-bcd(){  cd $(openmeshtest-bdir); }


openmeshtest-wipe(){
  local bdir=$(openmeshtest-bdir)
  rm -rf $bdir 

}

openmeshtest-cmake(){
  local iwd=$PWD
  local bdir=$(openmeshtest-bdir)
  mkdir -p $bdir
  openmeshtest-bcd

  cmake $(openmeshtest-dir) \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_INSTALL_PREFIX=$(openmeshtest-idir) 

  cd $iwd
}

openmeshtest-make(){
  local iwd=$PWD
  openmeshtest-bcd
  make $*
  cd $iwd
}

openmeshtest-install(){
  openmeshtest-make install
}

openmeshtest--(){
  openmeshtest-cmake
  openmeshtest-make
  openmeshtest-install
}

