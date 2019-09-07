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

ggeo-rel(){      echo ggeo ; }
ggeo-src(){      echo ggeo/ggeo.bash ; }
ggeo-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(ggeo-src)} ; }
ggeo-vi(){       vi $(ggeo-source) ; }
ggeo-usage(){ cat << \EOU


EOU
}

ggeo-env(){      olocal- ; opticks- ; }

ggeo-idir(){ echo $(opticks-idir); } 
ggeo-bdir(){ echo $(opticks-bdir)/$(ggeo-rel) ; }  

ggeo-sdir(){ echo $(opticks-home)/ggeo ; }
ggeo-tdir(){ echo $(opticks-home)/ggeo/tests ; }

ggeo-icd(){  cd $(ggeo-idir); }
ggeo-bcd(){  cd $(ggeo-bdir); }
ggeo-scd(){  cd $(ggeo-sdir)/$1; }
ggeo-tcd(){  cd $(ggeo-tdir) ; }
ggeo-cd(){  cd $(ggeo-sdir); }
ggeo-c(){   cd $(ggeo-sdir); }

ggeo-wipe(){
    local bdir=$(ggeo-bdir)
    rm -rf $bdir
}

ggeo-name(){ echo GGeo ; }
ggeo-tag(){  echo GGEO ; }

ggeo-apihh(){  echo $(ggeo-sdir)/$(ggeo-tag)_API_EXPORT.hh ; }
ggeo---(){     touch $(ggeo-apihh) ; ggeo--  ; }

ggeo--(){                   opticks-- $(ggeo-bdir) ; }
ggeo-t(){                   opticks-t $(ggeo-bdir) $* ; }

ggeo-genproj() { ggeo-scd ; opticks-genproj $(ggeo-name) $(ggeo-tag) ; }
ggeo-gentest() { ggeo-tcd ; opticks-gentest ${1:-GExample} $(ggeo-tag) ; }
ggeo-txt(){ vi $(ggeo-sdir)/CMakeLists.txt $(ggeo-tdir)/CMakeLists.txt ; }

   
ggeo-sln(){ echo $(ggeo-bdir)/$(ggeo-name).sln ; }
ggeo-slnw(){ vs- ; echo $(vs-wp $(ggeo-sln)) ; }
ggeo-vs(){  opticks-vs $(ggeo-sln) ; }


ggeo-nocache-test()
{
    local dir=/usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.36fd07b60ec7753c091b38b3f12b4389.dae/
    [ -d "$dir" ] && rm -rf $dir 

    OPTICKS_QUERY="range:0:1" lldb GMaterialLibTest 
}




ggeo-t-nocache-(){ cat << EON  
GMaterialLibTest
GScintillatorLibTest
GBndLibTest
GBndLibInitTest
GPartsTest
GPmtTest
BoundariesNPYTest
GAttrSeqTest
GGeoLibTest
GGeoTest
GMakerTest
NLookupTest
RecordsNPYTest 
GSceneTest
GMeshLibTest
EON
}       # these tests all fail when there is no geocache


ggeo-t-nocache-rmcache()
{
    ## NB this dir corresponds to OPTICKS_QUERY="range:0:1"
    local dir=/usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.36fd07b60ec7753c091b38b3f12b4389.dae/
    [ -d "$dir" ] && rm -rf $dir 
}

ggeo-t-nocache()
{
    local msg="=== $FUNCNAME :"
    ggeo-t-nocache-rmcache
    local testname
    $FUNCNAME- | while read testname ; do  
        echo 
        echo
        echo $msg $testname
        OPTICKS_QUERY="range:0:1" $testname
    done    
}

ggeo-t-nocache-lldb()
{   
    local testname=${1:-GMeshLibTest}
    ggeo-t-nocache-rmcache
    OPTICKS_QUERY="range:0:1" lldb $testname
}



