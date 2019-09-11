##
## Copyright (c) 2019 Opticks Team. All Rights Reserved.
##
## This file is part of Opticks
## (see https://bitbucket.org/simoncblyth/opticks).
##
## Licensed under the Apache License, Version 2.0 (the "License"); 
## you may not use this file except in compliance with the License.  
## You may obtain a copy of the License at
##
##   http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software 
## distributed under the License is distributed on an "AS IS" BASIS, 
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
## See the License for the specific language governing permissions and 
## limitations under the License.
##

okdist-source(){ echo $BASH_SOURCE ; }
okdist-vi(){ vi $(okdist-source) $(opticks-home)/notes/issues/packaging-opticks-and-externals-for-use-on-gpu-cluster.rst ; }
okdist-env(){ echo -n ; }
okdist-usage(){  cat << \EOU

Opticks Binary Distribution : create tarball for explosion on cvmfs 
=====================================================================

Dev Notes
----------

* notes/issues/packaging-opticks-and-externals-for-use-on-gpu-cluster.rst

EOU
}



okdist-version(){ echo 0.1.0 ; }
okdist-name(){    echo Opticks-$(okdist-version) ; }
okdist-ext(){     echo .tar ; }  

okdist-prefix(){  echo $(okdist-name)/$(opticks-okdist-dirlabel) ; }   ## hmm wrong way around ?
okdist-file(){    echo $(okdist-name)$(okdist-ext) ; }

okdist-path(){    echo $(opticks-dir)/$(okdist-file) ; }    
okdist-ls(){      local p=$(okdist-path) ; ls -l $p ; du -h $p ; }
okdist-tmp(){     echo /tmp/$USER/opticks/okdist-test ; }
okdist-cd(){      cd $(okdist-tmp) ; }
okdist-untar(){    
    local msg="=== $FUNCNAME :"
    local tmp=$(okdist-tmp)
    rm -rf $tmp
    mkdir -p $tmp
    local dist=$(okdist-path)
    echo $msg explode tarball $dist into tmp $tmp
    case $(okdist-ext) in 
       .tar.gz) tar zxvf $dist --strip 2 -C $tmp  ;;
          .tar) tar  xvf $dist --strip 2 -C $tmp  ;;
    esac
}
okdist-test-notes(){ cat << EON

* ppm snaps defaulting to triangulated, still need --xanalytic switch.
* how to be sure are using the packaged libs ?

  * recall some RPATH setup for Linux using ORIGIN which enables the
    libs to be found relative to the executables 
  * notes/issues/packaging-opticks-and-externals-for-use-on-gpu-cluster.rst  

  * where is the sensitivity to opticks-f OPTICKS_INSTALL_PREFIX

EON
}


okdist-snap-()
{
    PATH=$(okdist-tmp)/lib:$PATH which OpSnapTest   

}

okdist-test()
{
    #OPTICKS_INSTALL_PREFIX=$(okdist-tmp) $(okdist-tmp)/lib/OpticksResourceTest --envkey 
    #LD_TRACE_LOADED_OBJECTS=1 $(okdist-tmp)/lib/OpticksResourceTest

   # gdb --args \

    OPTICKS_INSTALL_PREFIX=$(okdist-tmp) \
        $(okdist-tmp)/lib/OpSnapTest --envkey --xanalytic --target 352851 --eye -1,-1,-1 --snapconfig "steps=10,eyestartz=-1,eyestopz=5" --size 2560,1440,1 --embedded

    local ppm=$(opticks-dir)/tmp/snap00000.ppm 
    ls -l $ppm
    open $ppm
}


okdist-t()
{
    okdist--         ## collect binaries into tarball

    okdist-untar     ## explode into tmp dir
    okdist-test      ## check usage of binaries
}

okdist--(){        
   opticks-
   opticks-cd       ## install directory 
   okdist.py --prefix $(okdist-prefix) --ext $(okdist-ext) 
   okdist-ls  
}

okdist-info(){ cat << EOI

   okdist-prefix : $(okdist-prefix)
   okdist-ext    : $(okdist-ext)
   okdist-tmp    : $(okdist-tmp)

EOI
}



