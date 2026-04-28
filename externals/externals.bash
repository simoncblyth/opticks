externals-base()
{
    : base of package prefixes that are not internally managed such as openssl and libcurl
    local base
    if [ -d "/usr/local/ExternalLibs" ]; then
        base=/usr/local/ExternalLibs
    else
        base=/tmp/$USER/ExternalLibs
    fi
    echo $base
}


externals-curl-notes(){ cat << EON

*externals-curl url*
    when EXTERNALS_DOWNLOAD_CACHE envvar is defined and EXTERNALS_DOWNLOAD_CACHE/dist
    exists where dist is the basename obtained from the url then the dist is
    copied to the pwd instead of being curled there

An example of EXTERNALS_DOWNLOAD_CACHE::

    /cvmfs/opticks.ihep.ac.cn/opticks_download_cache

EON
}

externals-curl(){
   local msg="=== $FUNCNAME :"
   local dir=$PWD
   local url=$1
   local dist=$(basename $url)
   local cmd
   if [ -z "$url" -o -z "$dist" ]; then
       cmd="echo $msg BAD url $url dist $dir"
   elif [ -n "$EXTERNALS_DOWNLOAD_CACHE" -a -f "$EXTERNALS_DOWNLOAD_CACHE/$dist" ]; then
       cmd="cp $EXTERNALS_DOWNLOAD_CACHE/$dist $dist"
   else
       cmd="curl -L -O $url"
   fi
   echo $msg dir $dir url $url dist $dist EXTERNALS_DOWNLOAD_CACHE $EXTERNALS_DOWNLOAD_CACHE cmd $cmd
   eval $cmd
}



externals-source(){ echo $BASH_SOURCE ; }
externals-vi(){     vi $BASH_SOURCE ; }
externals-dir(){    echo $(dirname $BASH_SOURCE) ; }

boost-(){            . $(externals-dir)/boost.bash             && boost-env $* ; }
ocmake-(){           . $(externals-dir)/ocmake.bash            && ocmake-env $* ; }
glm-(){              . $(externals-dir)/glm.bash               && glm-env $* ; }
plog-(){             . $(externals-dir)/plog.bash              && plog-env $* ; }
gleq-(){             . $(externals-dir)/gleq.bash              && gleq-env $* ; }
glfw-(){             . $(externals-dir)/glfw.bash              && glfw-env $* ; }
libcurl-(){          . $(externals-dir)/libcurl.bash           && libcurl-env $* ; }
openssl-(){          . $(externals-dir)/openssl.bash           && openssl-env $* ; }
glew-(){             . $(externals-dir)/glew.bash              && glew-env $* ; }
imgui-(){            . $(externals-dir)/imgui.bash             && imgui-env $* ; }
cuda-(){             . $(externals-dir)/cuda.bash              && cuda-env $* ; }
cudamac-(){          . $(externals-dir)/cudamac.bash           && cudamac-env $* ; }
cudalin-(){          . $(externals-dir)/cudalin.bash           && cudalin-env $* ; }
cu-(){               . $(externals-dir)/cuda.bash              && cuda-env $* ; }
optix-(){            . $(externals-dir)/optix.bash             && optix-env $* ; }
optix7-(){           . $(externals-dir)/optix7.bash            && optix7-env $* ; }
optix7sdk-(){        . $(externals-dir)/optix7sdk.bash         && optix7sdk-env $* ; }
optix7c-(){          . $(externals-dir)/optix7c.bash           && optix7c-env $* ; }
rcs-(){              . $(externals-dir)/rcs.bash               && rcs-env $* ; }
optixnote-(){        . $(externals-dir)/optixnote.bash         && optixnote-env $* ; }
xercesc-(){          . $(externals-dir)/xercesc.bash           && xercesc-env $* ; }
g4-(){               . $(externals-dir)/g4.bash                && g4-env $* ; }
clhep-(){            . $(externals-dir)/clhep.bash             && clhep-env $* ; }
bcm-(){              . $(externals-dir)/bcm.bash               && bcm-env $* ; }

thrust-(){           . $(externals-dir)/thrust.bash            && thrust-env $* ; }
cub-(){              . $(externals-dir)/cub.bash               && cub-env $* ; }
mgpu-(){             . $(externals-dir)/mgpu.bash              && mgpu-env $* ; }

g4dev-(){            . $(externals-dir)/g4dev.bash             && g4dev-env $* ; }
nljson-(){           . $(externals-dir)/nljson.bash            && nljson-env $* ; }
root-(){             . $(externals-dir)/root.bash              && root-env $* ; }

examples-(){         . $(opticks-home)/examples/examples.bash  && examples-env $* ; }


