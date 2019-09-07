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

okg4-source(){   echo $BASH_SOURCE ; }
okg4-vi(){       vi $(okg4-source) ; }
okg4-usage(){ cat << EOU

Integration of Opticks and Geant4 
===================================

EOU
}

okg4-env(){  
   olocal- 
   g4-
   opticks-
}



okg4-idir(){ echo $(opticks-idir); } 
okg4-bdir(){ echo $(opticks-bdir)/okg4 ; }
okg4-sdir(){ echo $(opticks-home)/okg4 ; }
okg4-tdir(){ echo $(opticks-home)/okg4/tests ; }

okg4-icd(){  cd $(okg4-idir); }
okg4-bcd(){  cd $(okg4-bdir); }
okg4-scd(){  cd $(okg4-sdir); }
okg4-tcd(){  cd $(okg4-tdir); }

okg4-dir(){  echo $(okg4-sdir) ; }
okg4-cd(){   cd $(okg4-dir); }
okg4-c(){   cd $(okg4-dir); }


okg4-name(){ echo okg4 ; }
okg4-tag(){  echo OKG4 ; }

okg4-apihh(){  echo $(okg4-sdir)/$(okg4-tag)_API_EXPORT.hh ; }
okg4---(){     touch $(okg4-apihh) ; okg4--  ; }



okg4-wipe(){    local bdir=$(okg4-bdir) ; rm -rf $bdir ; } 

okg4--(){       opticks-- $(okg4-bdir) ; } 
okg4-t(){       opticks-t $(okg4-bdir) $* ; } 
okg4-genproj(){ okg4-scd ; oks- ; oks-genproj $(okg4-name) $(okg4-tag) ; } 
okg4-gentest(){ okg4-tcd ; oks- ; oks-gentest ${1:-CExample} $(okg4-tag) ; } 
okg4-txt(){     vi $(okg4-sdir)/CMakeLists.txt $(okg4-tdir)/CMakeLists.txt ; } 



