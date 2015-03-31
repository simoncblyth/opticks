gleq-src(){      echo graphics/gleq/gleq.bash ; }
gleq-source(){   echo ${BASH_SOURCE:-$(env-home)/$(gleq-src)} ; }
gleq-vi(){       vi $(gleq-source) ; }
gleq-env(){      elocal- ; }
gleq-usage(){ cat << EOU





EOU
}
gleq-dir(){ echo $(local-base)/env/graphics/gleq ; }
gleq-sdir(){ echo $(env-home)/graphics/gleq ; }
gleq-cd(){  cd $(gleq-dir); }
gleq-scd(){  cd $(gleq-sdir); }
gleq-mate(){ mate $(gleq-dir) ; }
gleq-get(){
   local dir=$(dirname $(gleq-dir)) &&  mkdir -p $dir && cd $dir

   git clone https://github.com/elmindreda/gleq.git

}
gleq-hdr(){
   echo $(gleq-dir)/gleq.h
}


