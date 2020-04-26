#!/bin/bash -l
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


opticks-
oe-
om-

sdir=$(pwd)
name=$(basename $sdir)
bdir=/tmp/$USER/opticks/$name/build 

echo bdir $bdir name $name

rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 

  
om-cmake $sdir 
make
make install   

which $name
gdb $name


cat << EON > /dev/null

(lldb) bt
* thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x48)
  * frame #0: 0x000000010038c959 liboptix.1.dylib`___lldb_unnamed_symbol3828$$liboptix.1.dylib + 25
    frame #1: 0x000000010038cac1 liboptix.1.dylib`___lldb_unnamed_symbol3829$$liboptix.1.dylib + 33
    frame #2: 0x000000010038cbf2 liboptix.1.dylib`___lldb_unnamed_symbol3830$$liboptix.1.dylib + 34
    frame #3: 0x0000000100388d58 liboptix.1.dylib`___lldb_unnamed_symbol3779$$liboptix.1.dylib + 88
    frame #4: 0x00000001002de418 liboptix.1.dylib`___lldb_unnamed_symbol2943$$liboptix.1.dylib + 1192
    frame #5: 0x0000000100217169 liboptix.1.dylib`___lldb_unnamed_symbol1167$$liboptix.1.dylib + 1465
    frame #6: 0x0000000100216548 liboptix.1.dylib`___lldb_unnamed_symbol1163$$liboptix.1.dylib + 4040
    frame #7: 0x000000010021551e liboptix.1.dylib`___lldb_unnamed_symbol1162$$liboptix.1.dylib + 94
    frame #8: 0x00000001001d7dc8 liboptix.1.dylib`___lldb_unnamed_symbol788$$liboptix.1.dylib + 136
    frame #9: 0x0000000100142c53 liboptix.1.dylib`rtContextLaunch1D + 259
    frame #10: 0x000000010000558c UseOptiXProgramPP`optix::ContextObj::launch(this=0x0000000106503d00, entry_point_index=0, image_width=10) at optixpp_namespace.h:2531
    frame #11: 0x0000000100004b3b UseOptiXProgramPP`main(argc=1, argv=0x00007ffeefbff088) at UseOptiXProgramPP.cc:109
    frame #12: 0x00007fff7cad0015 libdyld.dylib`start + 1
    frame #13: 0x00007fff7cad0015 libdyld.dylib`start + 1
(lldb) f 11
frame #11: 0x0000000100004b3b UseOptiXProgramPP`main(argc=1, argv=0x00007ffeefbff088) at UseOptiXProgramPP.cc:109
   106 	    context->setRayGenerationProgram( entry_point_index, program ); 
   107 	
   108 	    unsigned width = 10u ; 
-> 109 	    context->launch( entry_point_index , width  ); 
   110 	
   111 	    LOG(info) << argv[0] ; 
   112 	
(lldb) 


EON


