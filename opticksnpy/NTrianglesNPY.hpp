#pragma once

#include "NGLM.hpp"

template <typename T> class NPY ;

struct nbbox ; 
struct ntriangle ; 

#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

class NPY_API NTrianglesNPY {
    public:
        static const glm::vec3 PX ; 
        static const glm::vec3 PY ; 
        static const glm::vec3 PZ ; 
        static const glm::vec3 MX ; 
        static const glm::vec3 MY ; 
        static const glm::vec3 MZ ; 
    public:
        static const glm::vec3 Ip0 ; 
        static const glm::vec3 Ip1 ; 
        static const glm::vec3 Ip2 ; 
        static const glm::vec3 Ip3 ; 
        static const glm::vec3 Ip4 ; 
        static const glm::vec3 Ip5 ;

        static const glm::vec3 Im0 ; 
        static const glm::vec3 Im1 ; 
        static const glm::vec3 Im2 ; 
        static const glm::vec3 Im3 ; 
        static const glm::vec3 Im4 ; 
        static const glm::vec3 Im5 ;
    public:
        static const glm::vec3 PXPYPZ ; 
        static const glm::vec3 PXPYMZ ; 
        static const glm::vec3 PXMYPZ ; 
        static const glm::vec3 PXMYMZ ; 
        static const glm::vec3 MXPYPZ ; 
        static const glm::vec3 MXPYMZ ; 
        static const glm::vec3 MXMYPZ ; 
        static const glm::vec3 MXMYMZ ; 
    public:
        static NTrianglesNPY* icosahedron();
        static NTrianglesNPY* hemi_octahedron();
        static NTrianglesNPY* octahedron();
        static NTrianglesNPY* cube();
        static NTrianglesNPY* box(const nbbox& bb);
        static NTrianglesNPY* sphere(unsigned int n_polar=24, unsigned int n_azimuthal=24); 
        static NTrianglesNPY* sphere(glm::vec4& param, unsigned int n_polar=24, unsigned int n_azimuthal=24); 
        static NTrianglesNPY* sphere(float ctmin, float ctmax, unsigned int n_polar=24, unsigned int n_azimuthal=24); 
    public:
        static NTrianglesNPY* disk(glm::vec4& param, unsigned int n_azimuthal=24); 
        static NTrianglesNPY* prism(const glm::vec4& param); 
    public:
        NTrianglesNPY();
        NTrianglesNPY(NPY<float>* tris, NPY<float>* normals=NULL);
    public:
        NTrianglesNPY* transform(glm::mat4& m);
        NTrianglesNPY* subdivide(unsigned int nsubdiv);
    public:
        void add(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, const glm::vec3& d);
        void add(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c);
        void add(const ntriangle& t );
        void add(NTrianglesNPY* other);
        void addNormal(const glm::vec3& n ); // in triplicate
        void addNormal(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c );
    public:
        NPY<float>* getBuffer();
        NPY<float>* getTris();
        NPY<float>* getNormals();
        unsigned int getNumTriangles();
        nbbox* findBBox();
        void setTransform(const glm::mat4& transform);
        void setTransform(const glm::vec3& scale, const glm::vec3& translate);
        glm::mat4 getTransform();
    private:
        NPY<float>*  m_tris ; 
        NPY<float>*  m_normals ; 
        glm::mat4    m_transform ; 
};

#include "NPY_TAIL.hh"


