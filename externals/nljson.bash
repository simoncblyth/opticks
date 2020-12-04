nljson-vi(){       vi $BASH_SOURCE ; }
nljson-usage(){ cat << EOU

* https://github.com/nlohmann/json
* https://github.com/nlohmann/json#json-as-first-class-data-type

EOU
}

nljson-env(){ olocal- ;  }
nljson-url(){ echo https://github.com/nlohmann/json/releases/download/v3.9.1/json.hpp ; }

nljson-prefix(){ echo $(opticks-prefix)/externals ; }
nljson-path(){   echo $(opticks-prefix)/externals/include/nljson/json.hpp ; }
nljson-get()
{
   local dir=$(dirname $(nljson-path)) &&  mkdir -p $dir && cd $dir
   [ ! -f json.hpp ] && curl -L -O $(nljson-url) 
}

nljson--(){
   nljson-get
}

nljson-r(){ vim -R $(nljson-path) ; }


