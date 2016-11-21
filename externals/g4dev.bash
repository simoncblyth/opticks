g4dev-src(){      echo externals/g4dev.bash ; }
g4dev-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(g4dev-src)} ; }
g4dev-vi(){       vi $(g4dev-source) ; }
g4dev-usage(){ cat << \EOU

Geant4 Development
=====================

The g4dev- functions are for working with versions of Geant4 other
than the current Opticks standard one. 

For the standard version use the g4- functions.




* https://github.com/Geant4/geant4/releases?after=v9.5.1


EOU
}
g4dev-env(){      
   olocal-  
   xercesc-  
   opticks-
}


g4dev-edir(){ echo $(opticks-home)/g4 ; }

g4dev-prefix(){ 
    case $NODE_TAG in 
       MGB) echo $HOME/local/opticks/externals ;;
         D) echo /usr/local/opticks/externals ;;
         X) echo /opt/geant4 ;;
         *) echo ${LOCAL_BASE:-/usr/local}/opticks/externals ;;
    esac
 }

g4dev-libsuffix(){ 
    case $NODE_TAG in 
         X) echo 64  ;;
         *) echo -n ;;
    esac
}



g4dev-tag(){   echo g4 ; }

# nom identifier needs to match the name of the folder created by exploding the zip or tarball
# unfortunamely this is not simply connected with the basename of the url
#g4dev-nom(){ echo Geant4-10.2.1 ; }
g4dev-nom(){ echo geant4-9.5.0 ; }

g4dev-url(){   
   case $(g4dev-nom) in
       Geant4-10.2.1) echo http://geant4.cern.ch/support/source/geant4_10_02_p01.zip ;;
        geant4-9.5.0) echo https://github.com/Geant4/geant4/archive/v9.5.0.zip ;;
   esac
}

g4dev-idir(){ echo $(g4dev-prefix) ; }
g4dev-dir(){   echo $(g4dev-prefix)/$(g4dev-tag)/$(g4dev-nom) ; } 

g4dev-dist(){ echo $(dirname $(g4dev-dir))/$(basename $(g4dev-url)) ; }
g4dev-filename(){  echo $(basename $(g4dev-url)) ; }
g4dev-name(){  local filename=$(g4dev-filename) ; echo ${filename%.*} ; }  
# hmm .tar.gz would still have a .tar on the name

g4dev-txt(){ vi $(g4dev-dir)/CMakeLists.txt ; }


g4dev-info(){  cat << EOI

    g4dev-nom  : $(g4dev-nom)
    g4dev-url  : $(g4dev-url)
    g4dev-dist : $(g4dev-dist)
    g4dev-filename : $(g4dev-filename)
    g4dev-name     : $(g4dev-name)


    g4dev-idir : $(g4dev-idir)
    g4dev-dir  : $(g4dev-dir)


EOI
}

g4dev-find(){ find $(g4dev-dir) -name ${1:-G4OpBoundaryProcess.cc} ; }


g4dev-bdir(){ echo $(g4dev-dir).build ; }

g4dev-cmake-dir(){     echo $(g4dev-prefix)/lib$(g4dev-libsuffix)/$(g4dev-nom) ; }
g4dev-examples-dir(){  echo $(g4dev-prefix)/share/$(g4dev-nom)/examples ; }


g4dev-ecd(){  cd $(g4dev-edir); }
g4dev-cd(){   cd $(g4dev-dir); }
g4dev-icd(){  cd $(g4dev-prefix); }
g4dev-bcd(){  cd $(g4dev-bdir); }
g4dev-ccd(){  cd $(g4dev-cmake-dir); }
g4dev-xcd(){  cd $(g4dev-examples-dir); }


g4dev-get-tgz(){
   local dir=$(dirname $(g4dev-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(g4dev-url)
   # replace zip to tar.gz
   url=${url/.zip/.tar.gz}
   local tgz=$(basename $url)
   local nam=${tgz/.tar.gz}

   [ ! -f "$tgz" ] && curl -L -O $url 
   [ ! -d "$nam" ] && tar zxvf $tgz 
}

g4dev-get(){
   local dir=$(dirname $(g4dev-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(g4dev-url)
   local dst=$(basename $url)
   local nom=$(g4dev-nom)

   [ ! -f "$dst" ] && curl -L -O $url 
   [ ! -d "$nom" ] && unzip $dst 
}

g4dev-wipe(){
   local bdir=$(g4dev-bdir)
   rm -rf $bdir
}



################# below funcions for styduing G4 source ##################################

g4dev-ifind(){ find $(g4dev-idir) -name ${1:-G4VUserActionInitialization.hh} ; }
g4dev-sfind(){ find $(g4dev-dir)/source -name ${1:-G4VUserActionInitialization.hh} ; }

g4dev-hh(){ find $(g4dev-dir)/source -name '*.hh' -exec grep -H ${1:-G4GammaConversion} {} \; ; }
g4dev-icc(){ find $(g4dev-dir)/source -name '*.icc' -exec grep -H ${1:-G4GammaConversion} {} \; ; }
g4dev-cc(){ find $(g4dev-dir)/source -name '*.cc' -exec grep -H ${1:-G4GammaConversion} {} \; ; }

g4dev-cls-copy(){
   local iwd=$PWD
   local name=${1:-G4Scintillation}
   local lname=${name/G4}

   local sauce=$(g4dev-dir)/source
   local hh=$(find $sauce -name "$name.hh")
   local cc=$(find $sauce -name "$name.cc")
   local icc=$(find $sauce -name "$name.icc")

   [ "$hh" != "" ]  && echo cp $hh $iwd/$lname.hh
   [ "$cc" != "" ] && echo cp $cc $iwd/$lname.cc
   [ "$icc" != "" ] && echo cp $icc $iwd/$lname.icc
}

g4dev-cls(){
   local iwd=$PWD
   g4dev-cd
   local name=${1:-G4Scintillation}

   local h=$(find source -name "$name.h")
   local hh=$(find source -name "$name.hh")
   local cc=$(find source -name "$name.cc")
   local icc=$(find source -name "$name.icc")

   local vcmd="vi -R $h $hh $icc $cc"
   echo $vcmd
   eval $vcmd

   cd $iwd
}

g4dev-look(){ 
   local iwd=$PWD
   g4dev-cd
   local spec=${1:-G4RunManagerKernel.cc:707}

   local name=${spec%:*}
   local line=${spec##*:}
   [ "$line" == "$spec" ] && line=1

   local fcmd="find source -name $name"
   local path=$($fcmd)

   echo $spec $name $line $path 

   if [ "$path" == "" ]; then 
      echo $msg FAILED to find $name with : $fcmd
      return 
   fi 
   local vcmd="vi -R $path +$line"
   echo $vcmd
   eval $vcmd
     
   cd $iwd
}


