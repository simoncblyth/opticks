# === func-gen- : graphics/assimp/AssimpTest/assimptest fgp graphics/assimp/AssimpTest/assimptest.bash fgn assimptest fgh graphics/assimp/AssimpTest
assimptest-src(){      echo graphics/assimp/AssimpTest/assimptest.bash ; }
assimptest-source(){   echo ${BASH_SOURCE:-$(env-home)/$(assimptest-src)} ; }
assimptest-vi(){       vi $(assimptest-source) ; }
assimptest-env(){      elocal- ; }
assimptest-usage(){ cat << EOU


RPATH not working 

::

    delta:AssimpTest blyth$ ./AssimpTest 
    dyld: Library not loaded: /usr/local/env/graphics//libassimp.3.dylib
      Referenced from: /usr/local/env/graphics/assimp/AssimpTest/./AssimpTest
      Reason: image not found
    Trace/BPT trap: 5
    delta:AssimpTest blyth$ DYLD_LIBRARY_PATH=/usr/local/env/graphics/lib ./AssimpTest 
    delta:AssimpTest blyth$ 


EOU
}
assimptest-bdir(){ echo $(local-base)/env/graphics/assimp/AssimpTest ; }
assimptest-sdir(){ echo $(env-home)/graphics/assimp/AssimpTest ; }
assimptest-bcd(){  cd $(assimptest-bdir); }
assimptest-scd(){  cd $(assimptest-sdir); }
assimptest-cd(){   cd $(assimptest-sdir); }


assimptest-cmake(){
   local iwd=$PWD
   local bdir=$(assimptest-bdir)
   mkdir -p $bdir
   assimptest-bcd

   cmake -DCMAKE_BUILD_TYPE=Debug $(assimptest-sdir)
   cd $iwd
}


assimptest-make(){
   local iwd=$PWD
   assimptest-bcd
   make $*
   cd $iwd
}

assimptest-run(){
  export-
  export-export
  assimp-
  local rc
  DYLD_LIBRARY_PATH=$(assimp-prefix)/lib $LLDB $(assimptest-bdir)/AssimpTestPP $*
  rc=$?
  echo rc $rc
}



assimptest--(){
  assimptest-make
  assimptest-run  $*
}

assimptest-out(){

  local out=$(assimptest-bdir)/out.txt
  if [ ! -f "$out" ]; then 
      assimptest-run unnamed > $out
  fi 

  vi $out

}





