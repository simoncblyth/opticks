# === func-gen- : g4/g4ex fgp g4/g4ex.bash fgn g4ex fgh g4
g4ex-src(){      echo g4/g4ex.bash ; }
g4ex-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(g4ex-src)} ; }
g4ex-vi(){       vi $(g4ex-source) ; }
g4ex-usage(){ cat << EOU


EOU
}

g4ex-env(){    
   olocal- 
   g4- 
  # g4-export

}

g4ex-name(){ echo basic/B1 ; }
#g4ex-name(){ echo extended/optical/LXe ; }
#g4ex-name(){ echo extended/optical/OpNovice ; }
#g4ex-name(){ echo extended/optical/wls ; }
#g4ex-name(){ echo extended/eventgenerator/exgps ; }
#g4ex-name(){ echo extended/eventgenerator/GunGPS ; }


g4ex-cd(){   cd $(g4ex-dir); }
g4ex-dir(){  echo $(g4-examples-dir)/$(g4ex-name) ; }
g4ex-bdir(){ echo $(local-base)/env/g4ex/$(g4ex-name).build   ; }
g4ex-idir(){ echo $(local-base)/env/g4ex/$(g4ex-name).install ; }
g4ex-prefix(){ echo $(local-base)/env/g4ex ; }


g4ex-exe-(){ echo $(g4ex-bdir)/$(g4ex-config)/${1}.exe ; }
g4ex-exename(){
   case $(g4ex-name) in 
                  "basic/B1") echo exampleB1 ;;
      "extended/optical/LXe") echo LXe ;;
   esac 
}
g4ex-exe(){ echo $(g4ex-exe- $(g4ex-exename)) ; }





g4ex-hhfind(){ find $(g4-examples-dir) -name '*.hh' -exec grep -H ${1:-Gun} {} \; ; }
g4ex-ccfind(){ find $(g4-examples-dir) -name '*.cc' -exec grep -H ${1:-Gun} {} \; ; }

g4ex-wipe(){
   local bdir=$(g4ex-bdir)
   rm -rf $bdir
}

g4ex-cmake(){
   local sdir=$(g4ex-dir)
   local bdir=$(g4ex-bdir)
   mkdir -p $bdir  
   cd $bdir


   cmake \
         -DGeant4_DIR=$(g4-cmake-dir) \
         -DWITH_GEANT4_UIVIS=OFF \
         -DCMAKE_INSTALL_PREFIX=$(g4ex-prefix) \
         $sdir
}

g4ex-configure()
{
   g4ex-wipe
   g4ex-cmake
}



#g4ex-config(){ echo Debug ; }
g4ex-config(){ echo RelWithDebInfo ; }

g4ex-make(){
   local bdir=$(g4ex-bdir)
   mkdir -p $bdir
   cd $bdir
   cmake --build . --config $(g4ex-config) --target ${1:-install}
}

g4ex-run(){
   local bdir=$(g4ex-bdir)
   cd $bdir
   case $(g4ex-name) in 
      "basic/B1") ./exampleB1 ;;
      "extended/optical/LXe") ./LXe ;;
   esac 

}

g4ex-srun(){

   g4ex-cd
   local exe=$
}



g4ex--(){
   g4ex-wipe
   g4ex-cmake
   g4ex-make
}




