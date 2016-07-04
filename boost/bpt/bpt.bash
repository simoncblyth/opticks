# === func-gen- : boost/bpt/bpt fgp boost/bpt/bpt.bash fgn bpt fgh boost/bpt
bpt-src(){      echo boost/bpt/bpt.bash ; }
bpt-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(bpt-src)} ; }
bpt-vi(){       vi $(bpt-source) ; }
bpt-usage(){ cat << EOU

Boost Property Tree : with XML/JSON/INI support
==================================================


* http://www.boost.org/doc/libs/1_58_0/doc/html/property_tree.html
* http://www.boost.org/doc/libs/1_58_0/doc/html/property_tree/accessing.html


EOU
}
bpt-dir(){ echo $(opticks-home)/boost/bpt ; }
bpt-cd(){  cd $(bpt-dir); }


bpt-name(){ echo BPT ; }

bpt-sdir(){ echo $(opticks-home)/boost/bpt ; }
bpt-idir(){ echo $(local-base)/env/boost/bpt ; }
bpt-bdir(){ echo $(bpt-idir).build ; }

bpt-scd(){  cd $(bpt-sdir); }
bpt-cd(){  cd $(bpt-sdir); }

bpt-icd(){  cd $(bpt-idir); }
bpt-bcd(){  cd $(bpt-bdir); }


bpt-wipe(){
   local bdir=$(bpt-bdir)
   rm -rf $bdir
}
bpt-env(){     
    olocal- 
}

bpt-cmake(){
   local iwd=$PWD

   local bdir=$(bpt-bdir)
   mkdir -p $bdir
  
   bpt-bcd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(bpt-idir) \
       $(bpt-sdir)

   cd $iwd
}

bpt-make(){
   local iwd=$PWD

   bpt-bcd 
   make $*

   cd $iwd
}

bpt-install(){
   bpt-make install
}

bpt-bin(){ echo $(bpt-idir)/bin/$(bpt-name) ; }
bpt-export()
{
   echo -n
} 

bpt-run(){ 
   local bin=$(bpt-bin)
   bpt-export
   $bin $*
}

bpt-lldb()
{
   local bin=$(bpt-bin)
   bpt-export
   lldb $bin -- $*
}

bpt--()
{
    bpt-wipe
    bpt-cmake
    bpt-make
    bpt-install
}


bpt-demo-(){ cat << EOD
<debug>
    <filename>debug.log</filename>
    <modules>
        <module>Finance</module>
        <module>Admin</module>
        <module>HR</module>
    </modules>
    <level>2</level>
</debug>
EOD
}

bpt-demo(){
   $FUNCNAME- > /tmp/bpt-demo.xml
}

