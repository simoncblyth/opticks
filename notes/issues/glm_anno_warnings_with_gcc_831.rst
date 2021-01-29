glm_anno_warnings_with_gcc_831
=================================


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

