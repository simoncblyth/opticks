g4x-source(){   echo $BASH_SOURCE ; }
g4x-vi(){       vi $(g4x-source) ; }
g4x-usage(){ cat << EOU


EOU
}

g4x-env(){  
   echo -n
}

g4x-name(){ echo ${G4X_NAME:-extended/optical/OpNovice} ; }
#g4x-name(){ echo ${G4X_NAME:-extended/optical/LXe} ; }

g4x-bin(){  echo $(g4x-idir)/bin/LXe ; }

g4x-curdir(){ echo $(g4-dir)/examples/$(g4x-name) ; }
g4x-devdir(){ echo $(g4dev-dir)/examples/$(g4x-name) ; }

g4x-dir(){  echo /tmp/$USER/opticks/examples/$(g4x-name) ; }
g4x-sdir(){ echo $(g4x-dir) ; }
g4x-bdir(){ echo $(g4x-dir).build ; }
g4x-idir(){ echo $(g4x-dir).install ; }

g4x-cd(){   cd $(g4x-dir); }
g4x-scd(){  cd $(g4x-sdir); }
g4x-bcd(){  cd $(g4x-bdir); }
g4x-icd(){  cd $(g4x-idir); }

g4x-get(){
   local dir=$(g4x-dir)
   local fold=$(dirname $dir)
   if [ ! -d $dir ]; then
       g4-
       mkdir -p $fold
       cp -r $(g4-examples-dir)/$(g4x-name) $fold/
   fi 
}

g4x-info(){ cat << EOI

   g4x-name : $(g4x-name)
   g4x-dir  : $(g4x-dir)

   g4x-sdir : $(g4x-sdir)
   g4x-bdir : $(g4x-bdir)
   g4x-idir : $(g4x-idir)

EOI
}

g4x-wipe(){
    local bdir=$(g4x-bdir)
    rm -rf $bdir
}

g4x-cmake(){
   local iwd=$PWD
   local bdir=$(g4x-bdir)
   mkdir -p $bdir
   g4x-bcd

  # -DWITH_GEANT4_UIVIS=OFF \

   cmake \
         -DGeant4_DIR=$(g4-cmake-dir) \
         -DCMAKE_INSTALL_PREFIX=$(g4x-idir) \
         -DCMAKE_BUILD_TYPE=Debug  \
         $(g4x-sdir)
   cd $iwd 
}

g4x-make(){
    local iwd=$PWD
    g4x-bcd
    make $*
    cd $iwd 
}

g4x-install(){
   g4x-make install
}

g4x--(){
   g4x-wipe
   g4x-cmake
   g4x-make
   g4x-make install
}

g4x-run(){
   local bin=$(g4x-bin)
   g4-export
   $bin $*
}


g4x-diff()
{
   cd $(opticks-prefix)/externals/g4
   g4-
   g4dev-
   
   cat << EOC
g4x-diff comparing::
  
   pwd        : $(pwd)
   g4-name    : $(g4-name)
   g4dev-name : $(g4dev-name)
   g4x-name   : $(g4x-name)

   diff -r --brief $(g4-name)/examples/$(g4x-name) $(g4dev-name)/examples/$(g4x-name)


   diff -r         $(g4-name)/examples/$(g4x-name) $(g4dev-name)/examples/$(g4x-name)
EOC

   diff -r --brief $(g4-name)/examples/$(g4x-name) $(g4dev-name)/examples/$(g4x-name)

   echo "\n\n\n"

   diff -r $(g4-name)/examples/$(g4x-name) $(g4dev-name)/examples/$(g4x-name)
}
