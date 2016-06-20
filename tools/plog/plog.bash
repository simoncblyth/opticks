# === func-gen- : tools/plog/plog fgp tools/plog/plog.bash fgn plog fgh tools/plog
plog-src(){      echo tools/plog/plog.bash ; }
plog-source(){   echo ${BASH_SOURCE:-$(env-home)/$(plog-src)} ; }
plog-vi(){       vi $(plog-source) ; }
plog-usage(){ cat << EOU




EOU
}
plog-env(){      opticks- ;  }
plog-dir(){ echo $(opticks-prefix)/externals/plog ; }
plog-cd(){  cd $(plog-dir); }
plog-url(){  echo https://github.com/SergiusTheBest/plog ; }
plog-get(){
   local dir=$(dirname $(plog-dir)) &&  mkdir -p $dir && cd $dir

   [ ! -d plog ] && git clone $(plog-url) 

}

plog-edit(){  vi $(opticks-home)/cmake/Modules/FindPLog.cmake ; }

