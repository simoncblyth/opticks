gleq-src(){      echo externals/gleq.bash ; }
gleq-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(gleq-src)} ; }
gleq-vi(){       vi $(gleq-source) ; }
gleq-env(){      olocal- ; opticks- ; }
gleq-usage(){ cat << EOU
GLEQ : Simple GLFW Event Queue 
=================================

* https://github.com/simoncblyth/gleq
* https://github.com/glfw/gleq

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
gleq-edit(){ vi $(opticks-home)/cmake/Modules/FindGLEQ.cmake ; }

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

