# === func-gen- : boost/bpt/bpt fgp boost/bpt/bpt.bash fgn bpt fgh boost/bpt
bpt-src(){      echo boost/bpt/bpt.bash ; }
bpt-source(){   echo ${BASH_SOURCE:-$(env-home)/$(bpt-src)} ; }
bpt-vi(){       vi $(bpt-source) ; }
bpt-env(){      elocal- ; }
bpt-usage(){ cat << EOU

Boost Property Tree : with XML/JSON/INI support
==================================================


* http://www.boost.org/doc/libs/1_58_0/doc/html/property_tree.html
* http://www.boost.org/doc/libs/1_58_0/doc/html/property_tree/accessing.html


EOU
}
bpt-dir(){ echo $(local-base)/env/boost/bpt/boost/bpt-bpt ; }
bpt-cd(){  cd $(bpt-dir); }
bpt-mate(){ mate $(bpt-dir) ; }
bpt-get(){
   local dir=$(dirname $(bpt-dir)) &&  mkdir -p $dir && cd $dir

}
