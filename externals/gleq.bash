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

gleq-src(){      echo externals/gleq.bash ; }
gleq-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(gleq-src)} ; }
gleq-vi(){       vi $(gleq-source) ; }
gleq-env(){      olocal- ; opticks- ; }
gleq-usage(){ cat << EOU
GLEQ : Simple GLFW Event Queue 
=================================

Overview
---------

* https://github.com/simoncblyth/gleq
* https://github.com/glfw/gleq

Given that gleq is just a single header, it appears that 
it is not handled as a usual external.  Instead the header
is just manually placed into oglrap/gleq.h.

Nevertheless for sanity, have updated my fork to 


New GLEQ
-----------

Updated GLEQ, March 22 2019


Old GLEQ
-----------

Used this old GLEQ from Sept 2015, from around then until March 2019.::

    [blyth@localhost gleq]$ git log
    commit 0b3ce0039995602a072198135618a0149f66630c
    Author: Camilla Berglund <elmindreda@elmindreda.org>
    Date:   Thu Sep 3 15:01:48 2015 +0200

        Add missing termination


Bringing the fork uptodate
-----------------------------

* https://help.github.com/en/articles/configuring-a-remote-for-a-fork

::

    [blyth@localhost gleq]$ git remote -v
    origin  https://github.com/simoncblyth/gleq (fetch)
    origin  https://github.com/simoncblyth/gleq (push)

    [blyth@localhost gleq]$ git remote add upstream https://github.com/glfw/gleq

    [blyth@localhost gleq]$ git remote -v
    origin  https://github.com/simoncblyth/gleq (fetch)
    origin  https://github.com/simoncblyth/gleq (push)
    upstream    https://github.com/glfw/gleq (fetch)
    upstream    https://github.com/glfw/gleq (push)
    [blyth@localhost gleq]$ 


* https://help.github.com/en/articles/syncing-a-fork


::

    [blyth@localhost gleq]$ git fetch upstream
    remote: Enumerating objects: 28, done.
    remote: Counting objects: 100% (28/28), done.
    remote: Total 75 (delta 28), reused 28 (delta 28), pack-reused 47
    Unpacking objects: 100% (75/75), done.
    From https://github.com/glfw/gleq
     * [new branch]      master     -> upstream/master


    [blyth@localhost gleq]$ l   ## working branch is still my old one
    total 24
    -rw-rw-r--. 1 blyth blyth 4647 Jul  5  2018 test.c
    -rw-rw-r--. 1 blyth blyth 8484 Jul  5  2018 gleq.h
    -rw-rw-r--. 1 blyth blyth 2736 Jul  5  2018 README.md
    [blyth@localhost gleq]$ 
    [blyth@localhost gleq]$ 
    [blyth@localhost gleq]$ git merge upstream/master    ## bring in the upstream
    Updating 0b3ce00..4dd5070
    Fast-forward
     LICENSE.md |  21 +++++++++++++++++++
     README.md  |  53 +++++++++++++++++++++++++++++++++-------------
     gleq.h     | 290 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++--------------------------------------------------------------------------------------
     test.c     | 129 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++--------------
     4 files changed, 364 insertions(+), 129 deletions(-)
     create mode 100644 LICENSE.md
    [blyth@localhost gleq]$ 


Pushing requested a password, despite ssh keys be setup already. Suspect 
this is due to using http url rather than the ssh one::

    blyth@localhost gleq]$ git push origin master
    Username for 'https://github.com': simoncblyth
    Password for 'https://simoncblyth@github.com': 
    Counting objects: 79, done.
    Delta compression using up to 48 threads.
    Compressing objects: 100% (75/75), done.
    Writing objects: 100% (75/75), 12.70 KiB | 0 bytes/s, done.
    Total 75 (delta 34), reused 0 (delta 0)
    remote: Resolving deltas: 100% (34/34), completed with 3 local objects.
    To https://github.com/simoncblyth/gleq
       0b3ce00..4dd5070  master -> master
    [blyth@localhost gleq]$ 





Opticks use of gleq.h
-----------------------

Via oglrap/GLEQ.hh which appears only in oglrap/Frame.cc::

    019 #include "Opticks.hh"
     20 
     21 
     22 #include <GL/glew.h>
     23 #include <GLFW/glfw3.h>
     24 
     25 #define GLEQ_IMPLEMENTATION
     26 #include "GLEQ.hh"
     27 
     28 #include "NGLM.hpp"
     29 
     30 #include "Frame.hh"
    ...
    274 void Frame::initContext()
    275 {
    276     // hookup the callbacks and arranges outcomes into event queue 
    277     gleqTrackWindow(m_window);
    278 
    ...
    399 void Frame::listen()
    400 {
    401     glfwPollEvents();
    402 
    403     GLEQevent event;
    404     while (gleqNextEvent(&event))
    405     {
    406         if(m_dumpevent) dump_event(event);
    407         handle_event(event);
    408         gleqFreeEvent(&event);
    409     }
    410 }
    ...
    631 void Frame::handle_event(GLEQevent& event)
    632 {
    633     // some events like key presses scrub the position 
    634     //m_pos_x = floor(event.pos.x);
    635     //m_pos_y = floor(event.pos.y);
    636     //printf("Frame::handle_event    %d %d    \n", m_pos_x, m_pos_y );
    637 
    638     switch (event.type)
    639     {
    640         case GLEQ_FRAMEBUFFER_RESIZED:
    641             // printf("Frame::handle_event framebuffer resized to (%i %i)\n", event.size.width, event.size.height);
    642             resize(event.size.width, event.size.height, m_coord2pixel);
    643             break;
    644         case GLEQ_WINDOW_MOVED:
    645         case GLEQ_WINDOW_RESIZED:
    646             // printf("Frame::handle_event window resized to (%i %i)\n", event.size.width, event.size.height);
    647             resize(event.size.width, event.size.height, m_coord2pixel);
    648             break;
    649         case GLEQ_WINDOW_CLOSED:
    ...
    732 void Frame::dump_event(GLEQevent& event)
    733 {
    734     switch (event.type)
    735     {
    736         case GLEQ_WINDOW_MOVED:
    737             printf("Window moved to (%.0f %.0f)\n", event.pos.x, event.pos.y);
    738             break;
    739         case GLEQ_WINDOW_RESIZED:
    740             printf("Window resized to (%i %i)\n", event.size.width, event.size.height);
    741             break;



Was also present in oglrap/GUI.cc but seems gleq symbols not used, so commented it out::

     01 #include "GUI.hh"
      2 
      3 #include <GL/glew.h>
      4 #include <GLFW/glfw3.h>
      5 
      6 //#include "GLEQ.hh"
      7 
      8 #include "BFile.hh"
      9 
     10 // npy-
     11 #include "Index.hpp"




Version Warning
----------------

::


    Scanning dependencies of target OpticksGL
    [ 16%] Building CXX object CMakeFiles/OpticksGL.dir/OKGLTracer.cc.o
    In file included from /home/blyth/local/opticks/include/OGLRap/GLEQ.hh:7:0,
                     from /home/blyth/local/opticks/include/OGLRap/Frame.hh:9,
                     from /home/blyth/opticks/opticksgl/OKGLTracer.cc:20:
    /home/blyth/local/opticks/include/OGLRap/gleq.h:37:6: warning: #warning "This version of GLEQ does not support events added after GLFW 3.1" [-Wcpp]
         #warning "This version of GLEQ does not support events added after GLFW 3.1"
          ^
    [ 33%] Linking CXX shared library libOpticksGL.so


Old? Observation
------------------

Hmm it appears gleq.h was just copied into oglrap, and slightly modified::

    epsilon:gleq blyth$ gleq-diff
    diff /usr/local/opticks/externals/gleq/gleq.h /Users/blyth/opticks/oglrap/gleq.h
    274c274
    <     event->file.paths = malloc(count * sizeof(char*));
    ---
    >     event->file.paths = (char**)malloc(count * sizeof(char*));
    epsilon:gleq blyth$ 


Compilation error from mountains- (which was porting from SFML to GLFW/GLEW/GLEQ)
presumably from compiling C as C++::

    make: *** [all] Error 2
    [ 20%] Building CXX object CMakeFiles/Mountains.dir/main.cpp.o
    In file included from /usr/local/env/graphics/opengl/mountains/main.cpp:17:
    In file included from /usr/local/env/graphics/opengl/mountains/GLEQ.hh:7:
    /usr/local/opticks/externals/gleq/gleq.h:274:23: error: assigning to 'char **' from incompatible type 'void *'
        event->file.paths = malloc(count * sizeof(char*));
                          ^ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1 error generated.

Fix easy, but why this has no effect in oglrap- ?::

    event->file.paths = (char**)malloc(count * sizeof(char*));


EOU
}
gleq-dir(){ echo $(opticks-prefix)/externals/gleq ; }
gleq-sdir(){ echo $(opticks-home)/graphics/gleq ; }
gleq-cd(){  cd $(gleq-dir); }
gleq-scd(){  cd $(gleq-sdir); }
gleq-edit(){ vi $(opticks-home)/cmake/Modules/FindGLEQ.cmake ; }  ## no such module, header just copied in ? 

gleq-url(){ echo https://github.com/simoncblyth/gleq ; }

gleq-get(){
   local iwd=$PWD
   local dir=$(dirname $(gleq-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d gleq ] && git clone $(gleq-url)
   cd $iwd
}
gleq-hdr(){
   echo $(gleq-dir)/gleq.h
}

gleq--(){
   gleq-get
}

gleq-diff()
{ 
    local cmd="diff $(gleq-dir)/gleq.h $(opticks-home)/oglrap/gleq.h" 
    echo $cmd
    eval $cmd
}

gleq-setup(){ cat << EOS
# $FUNCNAME  
EOS
}


