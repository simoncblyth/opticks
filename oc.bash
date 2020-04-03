oc-source(){ echo $BASH_SOURCE ; }
oc-vi(){ vi $(oc-source) ; }
oc-env(){  olocal- ; opticks- ; }

oc-pkg-config(){ PKG_CONFIG_PATH=$(opticks-prefix)/lib/pkgconfig:$(opticks-prefix)/externals/lib/pkgconfig pkg-config $* ; }




oc-usage(){ cat << EOU

OC : Opticks Config Hardcoded Minimal Approach 
===========================================================

* NB this aspires to becoming a standalone opticks-config script, 
  so keep use of other scripts to a minimum
  
    
EOU
} 

oc-prefix(){ echo $(opticks-prefix) ; }
oc-libdir(){ echo $(opticks-prefix)/lib ; }
oc-incdir(){ echo $(opticks-prefix)/include ; }
oc-extdir(){ echo $(opticks-prefix)/externals ; }

oc-deps-()
{
   : direct deps only 
   local pkg=$1 
   case $pkg in
          PLog) echo ;; 
           GLM) echo ;; 
        SysRap) echo PLog ;;   
      BoostRap) echo SysRap CPP ;; 
           NPY) echo BoostRap GLM ;; 
    YoctoGLRap) echo NPY ;; 
   OpticksCore) echo NPY ;; 
          GGeo) echo OpticksCore YoctoGLRap ;; 
   esac
}

oc-pkgs(){ cat << EOP
CPP
PLog
SysRap
BoostRap
NPY
GLM
YoctoGLRap
OpticksCore
GGeo
EOP
}

oc-libs-()
{
   : direct linker line of each package 
   local pkg=$1 
   case $pkg in
           CPP) printf "%s" -lstdc++ ;;    
      PLog|GLM) printf "" ;;

        SysRap) printf "%s " -l$pkg ;;   
      BoostRap) printf "%s " -l$pkg ;; 
           NPY) printf "%s " -l$pkg ;; 
          GGeo) printf "%s " -l$pkg ;; 
   esac
}

oc-cflags-()
{
   : direct cflags of each package 

   local pkg=$1 
   local msg="=== $FUNCNAME :"

   [ -n "$DEBUG" ] && echo $msg pkg $pkg 

   case $pkg in
          PLog) printf "%s " -I$(oc-extdir)/plog/include ;;
           GLM) printf "%s " -I$(oc-extdir)/glm/glm ;;  

        SysRap) printf "%s " -I$(oc-incdir)/$pkg ;;   
      BoostRap) printf "%s " -I$(oc-incdir)/$pkg ;; 
           NPY) printf "%s " -I$(oc-incdir)/$pkg ;; 
          GGeo) printf "%s " -I$(oc-incdir)/$pkg ;; 
   esac
}






oc-deps()
{
   : all deps determined recursively 

   local pkg=$1 
   local ddeps=$(oc-deps- $pkg)  # recurse over each of the direct dependencies

   for ddep in $ddeps ; do
      echo $ddep
      local deps=$(oc-deps $ddep)
      for dep in $deps ; do 
          echo $dep 
      done  
   done
}


oc-udeps(){ 
   local deps=$(oc-deps $*) 
   echo $deps | tr " " "\n" | sort | uniq
}



oc-libs()
{
   : only oc-deps is recursive this just converts the flat list of deps into libs

   local pkg=$1
   local deps=$(oc-deps $pkg)

   printf "%s " -L$(oc-libdir) 
   printf "%s " "$(oc-libs- $pkg)"

   for dep in $deps ; do
      printf "%s " "$(oc-libs- $dep)"
   done
   printf "\n" 
}

oc-cflags()
{
   : only oc-deps is recursive this just converts the flat list of deps into cflags

   local pkg=$1
   local deps=$(oc-deps $pkg)

   printf "%s " "$(oc-cflags- $pkg)"
   for dep in $deps ; do
      printf "%s " "$(oc-cflags- $dep)"
   done
   printf "\n" 
}

oc-dump()
{
   oc-pkgs | while read pkg ; do 
      printf "%10s %50s %50s \n" $pkg "$(oc-libs- $pkg)" "$(oc-cflags- $pkg)"  
   done

   oc-pkgs | while read pkg ; do 
      printf "%10s %50s %50s \n" $pkg "$(oc-libs $pkg)" "$(oc-cflags $pkg)"  
   done







}
