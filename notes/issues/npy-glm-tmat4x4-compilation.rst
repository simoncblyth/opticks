npy-glm-tmat4x4-compilation
===============================





Issue Reported by Yunlong
--------------------------

::

    Hi Simon,

    I'm a PhD student at the University of Manchester, working on the migration of
    Opticks to Geant4. I am trying to build the Opticks on a Centos 7 GPU machine.
    When I ran the "opticks-full" command after building all the foreign and
    external packages, it failed at the compilation of NPY package:

    ...
    In file included from /hepgpu6-data1/liyu/opticks/npy/NSpectral.cpp:22:0:
    /hepgpu6-data1/liyu/opticks/npy/GLMFormat.hpp:123:45: error: ‘tmat4x4’ in namespace ‘glm’ does not name a template type
     NPY_API std::string gfromstring_(const glm::tmat4x4<T>& m, bool flip=false) ;       
                                           ^
    /hepgpu6-data1/liyu/opticks/npy/GLMFormat.hpp:123:52: error: expected ‘,’ or ‘...’ before ‘<’ token
     NPY_API std::string gfromstring_(const glm::tmat4x4<T>& m, bool flip=false) ; 
    ....

    And these errors recurred where the types "const glm::tmat4x4" and "const
    glm::tvec4" are used. I didn't find the definitions of there types in the
    recommanded version of GLM package (0.9.9.5). So I tried to use earlier GLM
    package ( 0.9.8.5 or earlier) where these types are defined well, but the
    compilation of NPY package failed again with other errors:

    ...
    In file included from /hepgpu6-data1/liyu/opticks/npy/NScanLine.cpp:27:0:
    /hepgpu6-data1/liyu/opticks/npy/GLMFormat.hpp:138:75: error: wrong number of template arguments (1, should be 2)
     NPY_API std::string gpresent__(const char* label, const glm::tmat4x4<float>& m,  unsigned prec=3, unsigned wid=7, unsigned lwid=10, bool flip=false );
                                                                               ^
    In file included from /hepgpu6-data1/liyu/opticks.build/externals/glm/glm/glm/fwd.hpp:9:0,
                     from /hepgpu6-data1/liyu/opticks/npy/GLMFormat.hpp:22,
                     from /hepgpu6-data1/liyu/opticks/npy/NScanLine.cpp:27:
    /hepgpu6-data1/liyu/opticks.build/externals/glm/glm/glm/detail/type_mat.hpp:26:44: 
       error: provided for ‘template<class T, glm::precision P> struct glm::tmat4x4’
      template <typename T, precision P> struct tmat4x4;
    ...

    which said that wrong number of arguments are given to "strcut glm::tmat4x4".

    Could that be caused by my wrong installation of the Opticks or other reasons?

    Many thanks,
    Yunlong




Search for tmat4x4 in glm-0.9.9.5
-------------------------------------

::

    epsilon:glm blyth$ pwd
    /usr/local/opticks/externals/glm
    epsilon:glm blyth$ find glm-0.9.9.5 -name '*.hpp' -exec grep -H tmat4x4 {} \;
    glm-0.9.9.5/glm/glm/detail/qualifier.hpp:		template <typename T, qualifier Q = defaultp> using tmat4x4 = mat<4, 4, T, Q>;
    epsilon:glm blyth$ 

    038 
     39 #   if GLM_HAS_TEMPLATE_ALIASES
     40         template <typename T, qualifier Q = defaultp> using tvec1 = vec<1, T, Q>;
     41         template <typename T, qualifier Q = defaultp> using tvec2 = vec<2, T, Q>;
     42         template <typename T, qualifier Q = defaultp> using tvec3 = vec<3, T, Q>;
     43         template <typename T, qualifier Q = defaultp> using tvec4 = vec<4, T, Q>;
     44         template <typename T, qualifier Q = defaultp> using tmat2x2 = mat<2, 2, T, Q>;
     45         template <typename T, qualifier Q = defaultp> using tmat2x3 = mat<2, 3, T, Q>;
     46         template <typename T, qualifier Q = defaultp> using tmat2x4 = mat<2, 4, T, Q>;
     47         template <typename T, qualifier Q = defaultp> using tmat3x2 = mat<3, 2, T, Q>;
     48         template <typename T, qualifier Q = defaultp> using tmat3x3 = mat<3, 3, T, Q>;
     49         template <typename T, qualifier Q = defaultp> using tmat3x4 = mat<3, 4, T, Q>;
     50         template <typename T, qualifier Q = defaultp> using tmat4x2 = mat<4, 2, T, Q>;
     51         template <typename T, qualifier Q = defaultp> using tmat4x3 = mat<4, 3, T, Q>;
     52         template <typename T, qualifier Q = defaultp> using tmat4x4 = mat<4, 4, T, Q>;
     53         template <typename T, qualifier Q = defaultp> using tquat = qua<T, Q>;
     54 #   endif
     55 


    epsilon:glm blyth$ find glm-0.9.9.5 -name '*.hpp' -exec grep -H GLM_HAS_TEMPLATE_ALIASES {} \;
    glm-0.9.9.5/glm/glm/detail/qualifier.hpp:#	if GLM_HAS_TEMPLATE_ALIASES
    glm-0.9.9.5/glm/glm/detail/setup.hpp:#	define GLM_HAS_TEMPLATE_ALIASES __has_feature(cxx_alias_templates)
    glm-0.9.9.5/glm/glm/detail/setup.hpp:#	define GLM_HAS_TEMPLATE_ALIASES 1
    glm-0.9.9.5/glm/glm/detail/setup.hpp:#	define GLM_HAS_TEMPLATE_ALIASES ((GLM_LANG & GLM_LANG_CXX0X_FLAG) && (\
    glm-0.9.9.5/glm/glm/ext.hpp:#if GLM_HAS_TEMPLATE_ALIASES
    epsilon:glm blyth$ 


    0239 
     240 // N2258 http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2258.pdf
     241 #if GLM_COMPILER & GLM_COMPILER_CLANG
     242 #   define GLM_HAS_TEMPLATE_ALIASES __has_feature(cxx_alias_templates)
     243 #elif GLM_LANG & GLM_LANG_CXX11_FLAG
     244 #   define GLM_HAS_TEMPLATE_ALIASES 1
     245 #else
     246 #   define GLM_HAS_TEMPLATE_ALIASES ((GLM_LANG & GLM_LANG_CXX0X_FLAG) && (\
     247         ((GLM_COMPILER & GLM_COMPILER_INTEL)) || \
     248         ((GLM_COMPILER & GLM_COMPILER_VC) && (GLM_COMPILER >= GLM_COMPILER_VC12)) || \
     249         ((GLM_COMPILER & GLM_COMPILER_CUDA))))
     250 #endif
     251 

