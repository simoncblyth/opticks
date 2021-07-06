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
nljson-dist(){   echo $(nljson-path) ; }
nljson-get()
{
   local msg="=== $FUNCNAME :"
   local dir=$(dirname $(nljson-path)) &&  mkdir -p $dir && cd $dir

   local url=$(nljson-url)
   local name=$(basename $url)

   [ ! -s "$name" ] && opticks-curl $url 
   [ ! -s "$name" ] && echo $msg FAILED TO DOWNLOAD $name
   [ -s "$name" ]   # set rc 
}

nljson--(){
   nljson-get
   nljson-pc
}

nljson-r(){ vim -R $(nljson-path) ; }


nljson-pc-(){ cat << EOP

prefix=$(opticks-prefix)
includedir=\${prefix}/externals/include/nljson

Name: NLJSON
Description: JSON parse
Version: 0.1.0

Cflags:  -I\${includedir}
Libs: -lstdc++
Requires: 

EOP
}

nljson-pc-path(){ echo $(opticks-prefix)/externals/lib/pkgconfig/NLJSON.pc ; }
nljson-pc(){ 
   local msg="=== $FUNCNAME :"
   local path=$(nljson-pc-path)
   local dir=$(dirname $path)

   [ ! -d "$dir" ] && echo $msg creating dir $dir && mkdir -p $dir 
   echo $msg $path
   nljson-pc- > $path 
}



