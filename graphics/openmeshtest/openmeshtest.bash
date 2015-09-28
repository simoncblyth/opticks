# === func-gen- : graphics/openmeshtest/openmeshtest fgp graphics/openmeshtest/openmeshtest.bash fgn openmeshtest fgh graphics/openmeshtest
openmeshtest-src(){      echo graphics/openmeshtest/openmeshtest.bash ; }
openmeshtest-source(){   echo ${BASH_SOURCE:-$(env-home)/$(openmeshtest-src)} ; }
openmeshtest-vi(){       vi $(openmeshtest-source) ; }
openmeshtest-env(){      elocal- ; }
openmeshtest-usage(){ cat << EOU





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

