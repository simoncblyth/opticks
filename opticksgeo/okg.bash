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

okg-src(){      echo opticksgeo/okg.bash ; }
okg-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(okg-src)} ; }
okg-vi(){       vi $(okg-source) ; }
okg-usage(){ cat << EOU



EOU
}

okg-env(){      olocal- ; opticks- ; }

okg-dir(){  echo $(opticks-home)/opticksgeo ; }
okg-sdir(){ echo $(opticks-home)/opticksgeo ; }
okg-tdir(){ echo $(opticks-home)/opticksgeo/tests ; }
okg-idir(){ echo $(opticks-idir); } 
okg-bdir(){ echo $(opticks-bdir)/opticksgeo ; }  

okg-c(){    cd $(okg-dir); }
okg-cd(){   cd $(okg-dir); }
okg-icd(){  cd $(okg-idir); }
okg-bcd(){  cd $(okg-bdir); }
okg-scd(){  cd $(okg-sdir); }
okg-tcd(){  cd $(okg-tdir); }

okg-wipe(){ local bdir=$(okg-bdir) ; rm -rf $bdir ; }

okg-name(){ echo OpticksGeometry ; }
okg-tag(){  echo OKGEO ; }

okg-apihh(){  echo $(okg-sdir)/$(okg-tag)_API_EXPORT.hh ; }
okg---(){     touch $(okg-apihh) ; okg--  ; }


okg--(){        opticks--     $(okg-bdir) ; }
okg-t(){        opticks-t  $(okg-bdir) $* ; }
okg-clean(){    opticks-make- $(okg-bdir) clean ; }
okg-tl(){       opticks-tl $(okg-bdir) $* ; }

okg-genproj(){  okg-scd ; opticks-genproj $(okg-name) $(okg-tag) ; }
okg-gentest(){  okg-tcd ; opticks-gentest ${1:-OpticksGeometry} $(okg-tag) ; }

okg-txt(){   vi $(okg-sdir)/CMakeLists.txt $(okg-tdir)/CMakeLists.txt ; }


