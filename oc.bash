oc-source(){ echo $BASH_SOURCE ; }
oc-vi(){ vi $(oc-source) ; }
oc-env(){  olocal- ; opticks- ; }
oc-usage(){ cat << EOU

OC : Opticks Config Based on pkg-config
===========================================================

* NB this aspires to becoming a standalone opticks-config script, 
  so keep use of other scripts to a minimum


TODO 
-----

Avoid manual edits of::

   externals/lib/pkgconfig/assimp.pc



Externals require addition of INTERFACE_PKG_CONFIG_NAME in cmake/modules/FindName.cmake 
-------------------------------------------------------------------------------------------

* name corresponding to pkg-config pc file eg glm.pc, assimp.pc 

::

     39     set_target_properties(${_tgt} PROPERTIES
     40         INTERFACE_INCLUDE_DIRECTORIES "${GLM_INCLUDE_DIR}"
     41         INTERFACE_PKG_CONFIG_NAME "glm"
     42     )

* after adding that, need to rebuild packages that use what was found 



    
EOU
} 


#oc-pkg-config(){ PKG_CONFIG_PATH=$(opticks-prefix)/lib/pkgconfig:$(opticks-prefix)/externals/lib/pkgconfig pkg-config $* ; }
oc-pkg-config(){ PKG_CONFIG_PATH=$(opticks-prefix)/lib/pkgconfig:$(opticks-prefix)/xlib/pkgconfig pkg-config $* ; }
oc-libpath(){ echo $(opticks-prefix)/lib:$(opticks-prefix)/externals/lib ; }


# "public" interface
oc-cflags(){ oc-pkg-config $1 --cflags --define-prefix ; }
oc-libs(){   oc-pkg-config $1 --libs   --define-prefix ; }
oc-dump(){   oc-pkg-config-dump $1 ; }



oc-pkg-config-find(){
   local pkg=${1:-NPY}
   local lpkg=$(echo $pkg | tr [A-Z] [a-z])

   local ipc=$(opticks-prefix)/lib/pkgconfig/${lpkg}.pc
   local xpc=$(opticks-prefix)/xlib/pkgconfig/${lpkg}.pc

   if [ -f "$ipc" ]; then
      echo $ipc
   elif [ -f "$xpc" ]; then
      echo $xpc
   else
      echo $FUNCNAME failed for pkg $pkg ipc $ipc xpc $xpc
   fi 
}

oc-pkg-config-dump(){
   local pkg=${1:-NPY}
   local opt
   $FUNCNAME-opts- | while read opt ; do 
       cmd="oc-pkg-config $pkg $opt"  
       printf "\n\n# %s\n\n"  "$cmd"
       $cmd | tr " " "\n"
   done
}

oc-pkg-config-dump-opts-(){ cat << EOC
--print-requires --define-prefix
--cflags 
--cflags --define-prefix
--libs 
--libs --define-prefix
--cflags-only-I --define-prefix
EOC
}

oc-pkg-config-check-dirs(){
   local pkg=${1:-NPY}
   local line 
   local dir
   local exists 
   oc-pkg-config $pkg --cflags-only-I --define-prefix | tr " " "\n" | while read line ; do 
     dir=${line:2} 
     [ -d "$dir" ] && exists="Y" || exists="N"  
     printf " %s : %s \n" $exists $dir 
   done 
}



