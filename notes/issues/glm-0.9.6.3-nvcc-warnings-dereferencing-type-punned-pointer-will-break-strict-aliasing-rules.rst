glm-0.9.6.3-nvcc-warnings-dereferencing-type-punned-pointer-will-break-strict-aliasing-rules : FIXED
=========================================================================================================

overview : avoided by updating to glm 0.9.9.5
-----------------------------------------------

* New warnings with cuda 10.1 nvcc and glm 0.9.6.3
* Suspect only shows up with cuda code because of "-O2" being used there. 
* switching off strict-aliasing warnings makes it go away 

* https://blog.qt.io/blog/2011/06/10/type-punning-and-strict-aliasing/
* https://github.com/patriciogonzalezvivo/glslViewer/issues/55

These messages are a bug in GLM that was fixed in release 0.9.7.1.

* https://github.com/g-truc/glm/issues/370

See also

* thrap-glm-test-
* glm-nvcc-

Avoided this by updating to glm-0.9.9.5



nvcc from CUDA 10.1 has some new warnings with glm
-------------------------------------------------------

::

    calhost thrustrap]$ om-make
    === om-make-one : thrustrap       /home/blyth/opticks/thrustrap                                /home/blyth/local/opticks/build/thrustrap                    
    [  2%] Building NVCC (Device) object CMakeFiles/ThrustRap.dir/ThrustRap_generated_TUtil_.cu.o
    [  8%] Building NVCC (Device) object CMakeFiles/ThrustRap.dir/ThrustRap_generated_TBufPair_.cu.o
    [  8%] Building NVCC (Device) object CMakeFiles/ThrustRap.dir/ThrustRap_generated_TBuf_.cu.o
    [ 10%] Building NVCC (Device) object CMakeFiles/ThrustRap.dir/ThrustRap_generated_TSparse_.cu.o
    [ 13%] Building NVCC (Device) object CMakeFiles/ThrustRap.dir/ThrustRap_generated_TRngBuf_.cu.o
    /home/blyth/local/opticks/externals/glm/glm/glm/detail/func_packing.inl: In function ‘glm::uint glm::packUnorm2x16(const vec2&)’:
    /home/blyth/local/opticks/externals/glm/glm/glm/detail/func_packing.inl:42:46: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
       return reinterpret_cast<uint const &>(Topack);
                                                  ^
    /home/blyth/local/opticks/externals/glm/glm/glm/detail/func_packing.inl: In function ‘glm::vec2 glm::unpackUnorm2x16(glm::uint)’:
    /home/blyth/local/opticks/externals/glm/glm/glm/detail/func_packing.inl:47:49: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
       vec2 Unpack(reinterpret_cast<u16vec2 const &>(p));



isolated the issue with thrap-glm-warning glm-nvcc
----------------------------------------------------

Capture it with thrap-glm-warning::

    [blyth@localhost glm-nvcc]$ thrap-glm-warning | sh 
/home/blyth/local/opticks/externals/glm/glm/glm/detail/func_packing.inl: In function ‘glm::uint glm::packUnorm2x16(const vec2&)’:
/home/blyth/local/opticks/externals/glm/glm/glm/detail/func_packing.inl:42:46: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
   return reinterpret_cast<uint const &>(Topack);
                                              ^
/home/blyth/local/opticks/externals/glm/glm/glm/detail/func_packing.inl: In function ‘glm::vec2 glm::unpackUnorm2x16(glm::uint)’:
/home/blyth/local/opticks/externals/glm/glm/glm/detail/func_packing.inl:47:49: warning: dereferencing type-punned pointer will break strict-aliasing rules [-Wstrict-aliasing]
   vec2 Unpack(reinterpret_cast<u16vec2 const &>(p));
                                                
                                                     ^

