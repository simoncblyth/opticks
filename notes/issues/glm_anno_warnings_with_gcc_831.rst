glm_anno_warnings_with_gcc_831
=================================

Suppressed this warning using::

   -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored 

* see glm-nvcc-test 


Many of these::


    /home/simon/local/opticks/externals/glm/glm/glm/./ext/../detail/.././ext/../detail/type_mat3x4.hpp(36): warning: __host__ annotation is ignored on a function("mat") that is explicitly defaulted on its first declaration

    /home/simon/local/opticks/externals/glm/glm/glm/./ext/../detail/.././ext/../detail/type_mat4x2.hpp(36): warning: __device__ annotation is ignored on a function("mat") that is explicitly defaulted on its first declaration

    /home/simon/local/opticks/externals/glm/glm/glm/./ext/../detail/.././ext/../detail/type_mat4x2.hpp(36): warning: __host__ annotation is ignored on a function("mat") that is explicitly defaulted on its first declaration


/home/simon/local/opticks/externals/glm/glm/glm/./ext/../detail/.././ext/../detail/type_mat3x4.hpp::

     25     public:
     26         // -- Accesses --
     27 
     28         typedef length_t length_type;
     29         GLM_FUNC_DECL static GLM_CONSTEXPR length_type length() { return 3; }
     30 
     31         GLM_FUNC_DECL col_type & operator[](length_type i);
     32         GLM_FUNC_DECL GLM_CONSTEXPR col_type const& operator[](length_type i) const;
     33 
     34         // -- Constructors --
     35 
     36         GLM_FUNC_DECL GLM_CONSTEXPR mat() GLM_DEFAULT;
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     37         template<qualifier P>
     38         GLM_FUNC_DECL GLM_CONSTEXPR mat(mat<3, 4, T, P> const& m);
     39 



::

    epsilon:glm-0.9.9.5 blyth$ find . -name '*.hpp' -exec grep -H GLM_DEFAULT {} \;
    ./glm/glm/detail/type_quat.hpp:		GLM_FUNC_DECL GLM_CONSTEXPR qua() GLM_DEFAULT;
    ./glm/glm/detail/type_quat.hpp:		GLM_FUNC_DECL GLM_CONSTEXPR qua(qua<T, Q> const& q) GLM_DEFAULT;
    ./glm/glm/detail/type_quat.hpp:		GLM_FUNC_DECL qua<T, Q>& operator=(qua<T, Q> const& q) GLM_DEFAULT;
    ./glm/glm/detail/type_mat3x3.hpp:		GLM_FUNC_DECL GLM_CONSTEXPR mat() GLM_DEFAULT;
    ./glm/glm/detail/type_mat3x2.hpp:		GLM_FUNC_DECL GLM_CONSTEXPR mat() GLM_DEFAULT;
    ./glm/glm/detail/setup.hpp:#	define GLM_DEFAULT = default
    ./glm/glm/detail/setup.hpp:#	define GLM_DEFAULT
    ./glm/glm/detail/type_mat3x4.hpp:		GLM_FUNC_DECL GLM_CONSTEXPR mat() GLM_DEFAULT;
    ./glm/glm/detail/type_mat2x3.hpp:		GLM_FUNC_DECL GLM_CONSTEXPR mat() GLM_DEFAULT;
    ./glm/glm/detail/type_mat4x4.hpp:		GLM_FUNC_DECL GLM_CONSTEXPR mat() GLM_DEFAULT;
    ./glm/glm/detail/type_mat2x2.hpp:		GLM_FUNC_DECL GLM_CONSTEXPR mat() GLM_DEFAULT;
    ./glm/glm/detail/type_vec1.hpp:		GLM_FUNC_DECL GLM_CONSTEXPR vec() GLM_DEFAULT;
    ./glm/glm/detail/type_vec1.hpp:		GLM_FUNC_DECL GLM_CONSTEXPR vec(vec const& v) GLM_DEFAULT;
    ./glm/glm/detail/type_vec1.hpp:		GLM_FUNC_DECL GLM_CONSTEXPR vec<1, T, Q> & operator=(vec const& v) GLM_DEFAULT;
    ./glm/glm/detail/type_vec3.hpp:		GLM_FUNC_DECL GLM_CONSTEXPR vec() GLM_DEFAULT;
    ./glm/glm/detail/type_vec3.hpp:		GLM_FUNC_DECL GLM_CONSTEXPR vec(vec const& v) GLM_DEFAULT;
    ./glm/glm/detail/type_vec3.hpp:		GLM_FUNC_DECL GLM_CONSTEXPR vec<3, T, Q>& operator=(vec<3, T, Q> const& v) GLM_DEFAULT;
    ./glm/glm/detail/type_vec2.hpp:		GLM_FUNC_DECL GLM_CONSTEXPR vec() GLM_DEFAULT;
    ./glm/glm/detail/type_vec2.hpp:		GLM_FUNC_DECL GLM_CONSTEXPR vec(vec const& v) GLM_DEFAULT;
    ./glm/glm/detail/type_vec2.hpp:		GLM_FUNC_DECL GLM_CONSTEXPR vec<2, T, Q> & operator=(vec const& v) GLM_DEFAULT;
    ./glm/glm/detail/type_mat4x3.hpp:		GLM_FUNC_DECL GLM_CONSTEXPR mat() GLM_DEFAULT;
    ./glm/glm/detail/type_mat4x2.hpp:		GLM_FUNC_DECL GLM_CONSTEXPR mat() GLM_DEFAULT;
    ./glm/glm/detail/type_mat2x4.hpp:		GLM_FUNC_DECL GLM_CONSTEXPR mat() GLM_DEFAULT;
    ./glm/glm/detail/type_vec4.hpp:		GLM_FUNC_DECL GLM_CONSTEXPR vec() GLM_DEFAULT;
    ./glm/glm/detail/type_vec4.hpp:		GLM_FUNC_DECL GLM_CONSTEXPR vec(vec<4, T, Q> const& v) GLM_DEFAULT;
    ./glm/glm/detail/type_vec4.hpp:		GLM_FUNC_DECL GLM_CONSTEXPR vec<4, T, Q>& operator=(vec<4, T, Q> const& v) GLM_DEFAULT;
    ./glm/glm/gtx/dual_quaternion.hpp:		GLM_FUNC_DECL GLM_CONSTEXPR tdualquat() GLM_DEFAULT;
    ./glm/glm/gtx/dual_quaternion.hpp:		GLM_FUNC_DECL GLM_CONSTEXPR tdualquat(tdualquat<T, Q> const& d) GLM_DEFAULT;
    ./glm/glm/gtx/dual_quaternion.hpp:		GLM_FUNC_DECL tdualquat<T, Q> & operator=(tdualquat<T, Q> const& m) GLM_DEFAULT;
    epsilon:glm-0.9.9.5 blyth$ 
    epsilon:glm-0.9.9.5 blyth$ 


./glm/glm/detail/setup.hpp::

     761 ///////////////////////////////////////////////////////////////////////////////////
     762 // Configure the use of defaulted function
     763 
     764 #if GLM_HAS_DEFAULTED_FUNCTIONS && GLM_CONFIG_CTOR_INIT == GLM_CTOR_INIT_DISABLE
     765 #   define GLM_CONFIG_DEFAULTED_FUNCTIONS GLM_ENABLE
     766 #   define GLM_DEFAULT = default
     767 #else
     768 #   define GLM_CONFIG_DEFAULTED_FUNCTIONS GLM_DISABLE
     769 #   define GLM_DEFAULT
     770 #endif
     771 



Issue from nvcc::



    [simon@localhost ~]$ ./t2.sh 
    /home/simon/local/opticks/externals/glm/glm/glm/./ext/../detail/type_vec2.hpp(94): warning: __device__ annotation is ignored on a function("vec") that is explicitly defaulted on its first declaration

    /home/simon/local/opticks/externals/glm/glm/glm/./ext/../detail/type_vec2.hpp(94): warning: __host__ annotation is ignored on a function("vec") that is explicitly defaulted on its first declaration

    /home/simon/local/opticks/externals/glm/glm/glm/./ext/../detail/type_vec2.hpp(95): warning: __device__ annotation is ignored on a function("vec") that is explicitly defaulted on its first declaration

    /home/simon/local/opticks/externals/glm/glm/glm/./ext/../detail/type_vec2.hpp(95): warning: __host__ annotation is ignored on a function("vec") that is explicitly defaulted on its first declaration

    ...  hundreds of lines like that 


    [simon@localhost ~]$ cat t2.sh 
    #!/bin/bash -l 


    /usr/local/cuda-10.1/bin/nvcc /home/simon/opticks/thrustrap/TCURANDImp.cu \
          -c -o /home/simon/local/opticks/build/thrustrap/CMakeFiles/ThrustRap.dir//./ThrustRap_generated_TCURANDImp.cu.o \
            -ccbin /opt/rh/devtoolset-8/root/usr/bin/cc -m64 \
            -DThrustRap_EXPORTS -DOPTICKS_THRAP -DOPTICKS_OKCORE -DOPTICKS_NPY -DOPTICKS_SYSRAP \
            -DOPTICKS_OKCONF -DOPTICKS_BRAP -DWITH_BOOST_ASIO -DBOOST_SYSTEM_DYN_LINK -DBOOST_ALL_NO_LIB \
            -DBOOST_PROGRAM_OPTIONS_DYN_LINK -DBOOST_FILESYSTEM_DYN_LINK -DBOOST_REGEX_DYN_LINK -DOPTICKS_CUDARAP \
    -Xcompiler ,\"-fvisibility=hidden\",\"-fvisibility-inlines-hidden\",\"-fdiagnostics-show-option\",\"-Wall\",\"-Wno-unused-function\",\"-Wno-comment\",\"-Wno-deprecated\",\"-Wno-shadow\",\"-fPIC\",\"-g\" \
     -Xcompiler -fPIC -gencode=arch=compute_70,code=sm_70 -std=c++11 -O2 --use_fast_math -DNVCC \
       -I/usr/local/cuda-10.1/include -I/home/simon/opticks/thrustrap -I/home/simon/local/opticks/include/OpticksCore -I/home/simon/local/opticks/externals/include -I/home/simon/local/opticks/include/NPY \
      -I/home/simon/local/opticks/externals/glm/glm -I/home/simon/local/opticks/include/SysRap -I/home/simon/local/opticks/externals/plog/include -I/home/simon/local/opticks/include/OKConf \
       -I/home/simon/local/opticks/include/BoostRap -I/home/simon/local/opticks_externals/boost/include -I/home/simon/local/opticks/externals/include/nljson \
        -I/home/simon/local/opticks/include/CUDARap -I/usr/local/cuda-10.1/samples/common/inc


    [simon@localhost ~]$ 


nvcc from CUDA 10.1 compilation of glm/glm.hpp (with gcc 8.3.1 underneath) spews hundreds of warnings::


    glm-nvcc-test-(){ cat << EOC
    #include <glm/glm.hpp>
    EOC
    }

    glm-nvcc-test(){
       : see notes/issues/glm_anno_warnings_with_gcc_831.rst 
       local tmpdir=/tmp/$USER/$FUNCNAME
       mkdir -p $tmpdir
       cd $tmpdir
       local capability=${OPTICKS_COMPUTE_CAPABILITY:-70} 

       $FUNCNAME- > tglm.cu

       local ccbin=$(which cc)
       echo ccbin $ccbin

       nvcc tglm.cu -c -ccbin $ccbin -m64 -I$(glm-dir2) \
      -Xcompiler ,\"-fvisibility=hidden\",\"-fvisibility-inlines-hidden\",\"-fdiagnostics-show-option\",\"-Wall\",\"-Wno-unused-function\",\"-Wno-comment\",\"-Wno-deprecated\",\"-Wno-shadow\",\"-fPIC\",\"-g\" \
      -Xcompiler -fPIC -gencode=arch=compute_$capability,code=sm_$capability -std=c++11 -O2 --use_fast_math -DNVCC

    }



* https://stackoverflow.com/questions/46469062/warning-host-annotation-on-a-defaulted-function-is-ignored-why

I think you probably should get a warning. Defaulting a constructor or
destructor is telling the compiler to generate its own trivial default
implementation automagically. Adding an annotation is irrelevant in this case.
Both compilers will generate a default, specifying that the default from either
host or device compiler should exist on both host and device is wrong in this
case. – talonmies Sep 28 '17 at 15:44


* http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2346.htm#trivial

NVIDIA claim that the device toolchain supports N2346 . If you want that
behaviour (and actually understand what it entails), then by all means use
defaulted constructors or destructors. But in that case specifying __host__
__device__ for a defaulted function doesn't make sense to me, and I think the
warning is valid. But what do I know.... – talonmies Sep 28 '17 at 16:08 


@Matthias: Very late follow up, but remember nvcc isn't a compiler. It runs all
code through two parallel compiler passes (host and device), and each compiler
will emit its own default implementation. The warning comes for exactly this
reason -- the device code compiler sees the __host__ decorator applied to a
device default and warns it is irrelevant at that point in the compilation
trajectory. – talonmies Aug 9 '18 at 8:32 



* https://github.com/kokkos/kokkos/issues/1473


::

   -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored


* https://stackoverflow.com/questions/14831051/how-to-disable-a-specific-nvcc-compiler-warnings
* http://www.ssl.berkeley.edu/~jimm/grizzly_docs/SSL/opt/intel/cc/9.0/lib/locale/en_US/mcpcom.msg

