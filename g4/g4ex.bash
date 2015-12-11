# === func-gen- : g4/g4ex fgp g4/g4ex.bash fgn g4ex fgh g4
g4ex-src(){      echo g4/g4ex.bash ; }
g4ex-source(){   echo ${BASH_SOURCE:-$(env-home)/$(g4ex-src)} ; }
g4ex-vi(){       vi $(g4ex-source) ; }
g4ex-usage(){ cat << EOU


EOU
}

g4ex-env(){    
   elocal- 
   g4- 
   g4-export

}

#g4ex-name(){ echo basic/B1 ; }
#g4ex-name(){ echo extended/optical/LXe ; }
g4ex-name(){ echo extended/optical/OpNovice ; }

g4ex-cd(){   cd $(g4ex-dir); }
g4ex-dir(){  echo $(g4-examples-dir)/$(g4ex-name) ; }
g4ex-bdir(){ echo $(local-base)/env/g4ex/$(g4ex-name).build   ; }
g4ex-idir(){ echo $(local-base)/env/g4ex/$(g4ex-name).install ; }

g4ex-wipe(){
   local bdir=$(g4ex-bdir)
   rm -rf $bdir
}

g4ex-cmake(){
   local sdir=$(g4ex-dir)
   local bdir=$(g4ex-bdir)
   mkdir -p $bdir  
   cd $bdir
   cmake -DGeant4_DIR=$(g4-cmake-dir) -DWITH_GEANT4_UIVIS=OFF $sdir
}

g4ex-make(){
   local bdir=$(g4ex-bdir)
   cd $bdir
   make -j4 VERBOSE=1
}

g4ex-run(){
   local bdir=$(g4ex-bdir)
   cd $bdir
   case $(g4ex-name) in 
      "basic/B1") ./exampleB1 ;;
      "extended/optical/LXe") ./LXe ;;
   esac 

}

g4ex--(){
   g4ex-wipe
   g4ex-cmake
   g4ex-make
}




