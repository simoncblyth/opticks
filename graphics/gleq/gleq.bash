gleq-src(){      echo graphics/gleq/gleq.bash ; }
gleq-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gleq-src)} ; }
gleq-vi(){       vi $(gleq-source) ; }
gleq-env(){      elocal- ; opticks- ; }
gleq-usage(){ cat << EOU



EOU
}
gleq-dir(){ echo $(opticks-prefix)/externals/gleq ; }
gleq-sdir(){ echo $(env-home)/graphics/gleq ; }
gleq-cd(){  cd $(gleq-dir); }
gleq-scd(){  cd $(gleq-sdir); }
gleq-mate(){ mate $(gleq-dir) ; }

gleq-get(){
   local iwd=$PWD
   local dir=$(dirname $(gleq-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d gleq ] && git clone https://github.com/simoncblyth/gleq
   cd $iwd
}
gleq-hdr(){
   echo $(gleq-dir)/gleq.h
}

gleq--(){
   gleq-get
}

