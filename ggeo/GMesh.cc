/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */


#include <cfloat>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <cassert>
#include <iomanip>
#include <algorithm>

#include <boost/filesystem.hpp>

#include "BFile.hh"
#include "BStr.hh"


#include "NGLM.hpp"
#include "NGLMExt.hpp"
#include "GLMFormat.hpp"
#include "NBBox.hpp"
#include "NPY.hpp"
#include "NCSG.hpp"


#include "GMatrix.hh"
#include "GMeshFixer.hh"
#include "GBuffer.hh"
#include "GMesh.hh"

#include "PLOG.hh"

const plog::Severity GMesh::LEVEL = PLOG::EnvLevel("GMesh", "DEBUG" ); 

namespace fs = boost::filesystem;


const char* GMesh::vertices_     = "vertices" ;
const char* GMesh::normals_      = "normals" ;
const char* GMesh::colors_       = "colors" ;
const char* GMesh::texcoords_    = "texcoords" ;

const char* GMesh::indices_      = "indices" ;
const char* GMesh::nodes_        = "nodes" ;
const char* GMesh::boundaries_   = "boundaries" ;
const char* GMesh::sensors_      = "sensors" ;

const char* GMesh::center_extent_ = "center_extent" ;
const char* GMesh::bbox_           = "bbox" ;
const char* GMesh::transforms_     = "transforms" ;
const char* GMesh::meshes_         = "meshes" ;
const char* GMesh::nodeinfo_       = "nodeinfo" ;
const char* GMesh::identity_       = "identity" ;


const char* GMesh::itransforms_    = "itransforms" ;
const char* GMesh::iidentity_       = "iidentity" ;

const char* GMesh::components_      = "components" ;


void GMesh::nameConstituents(std::vector<std::string>& names)
{
    names.push_back(vertices_); 
    names.push_back(normals_); 
    names.push_back(colors_); 
    names.push_back(texcoords_); 

    names.push_back(indices_); 
    names.push_back(nodes_); 
    names.push_back(boundaries_); 
    names.push_back(sensors_); 

    names.push_back(center_extent_); 
    names.push_back(bbox_); 
    names.push_back(transforms_); 
    names.push_back(meshes_); 
    names.push_back(nodeinfo_); 
    names.push_back(identity_); 

    names.push_back(itransforms_); 
    names.push_back(iidentity_); 

    names.push_back(components_); 
}


int GMesh::g_instance_count = 0 ;


bool GMesh::isEmpty() const 
{
    return m_num_vertices == 0 && m_num_faces == 0 ; 
}










GMesh::GMesh(unsigned int index, 
             gfloat3* vertices, 
             unsigned num_vertices, 
             guint3*  faces, 
             unsigned num_faces, 
             gfloat3* normals, 
             gfloat2* texcoords
            ) 
        :
      GDrawable(),
      m_index(index),
      m_globalinstance(false),

      m_num_vertices(num_vertices), 
      m_num_faces(num_faces),
      m_num_volumes(0),
      m_num_volumes_selected(0),
      m_num_mergedmesh(0),

      m_nodes(NULL),          
      m_boundaries(NULL),
      m_sensors(NULL),

      m_vertices(NULL),
      m_normals(NULL),
      m_colors(NULL),
      m_texcoords(NULL),
      m_faces(NULL),

      m_low(NULL),
      m_high(NULL),
      m_dimensions(NULL),
      m_center(NULL),
      m_extent(0.f),

      m_center_extent(NULL),
      m_bbox(NULL),
      m_transforms(NULL),
      m_itransforms(NULL),
      m_meshes(NULL),
      m_nodeinfo(NULL),
      m_identity(NULL),
      m_iidentity(NULL),

      m_model_to_world(NULL),
      m_name(NULL),
      m_shortname(NULL),
      m_version(NULL),
      m_geocode('T'),   // 'T'/'A' 
      m_islice(NULL),
      m_fslice(NULL),
      m_pslice(NULL),

      m_vertices_buffer(NULL),
      m_normals_buffer(NULL),
      m_colors_buffer(NULL),
      m_texcoords_buffer(NULL),
      m_indices_buffer(NULL),
      m_center_extent_buffer(NULL),
      m_bbox_buffer(NULL),
      m_nodes_buffer(NULL),
      m_boundaries_buffer(NULL),
      m_sensors_buffer(NULL),
      m_transforms_buffer(NULL),
      m_meshes_buffer(NULL),
      m_nodeinfo_buffer(NULL),
      m_identity_buffer(NULL),

      m_itransforms_buffer(NULL),
      m_iidentity_buffer(NULL),

      m_components_buffer(NULL),

      m_facerepeated_identity_buffer(NULL),
      m_facerepeated_iidentity_buffer(NULL),
      m_analytic_geometry_buffer(NULL),

      m_csg(NULL),
      m_alt(NULL),
      m_verbosity(0),
      m_parts(NULL),
      m_pt(NULL),

      m_x4src_vtx(NULL), 
      m_x4src_idx(NULL),
      m_g4vsolid(NULL)  

{
     init(vertices, faces, normals, texcoords);
}




void GMesh::stealIdentity(GMesh* other)
{
    setParts(other->getParts());
    setITransformsBuffer( other->getITransformsBuffer() );
    setInstancedIdentityBuffer( other->getInstancedIdentityBuffer() );

    // hmm passing everything over is too complicated, maybe better to do LODification inplace
    //     to avoid this        
}






void GMesh::setCSG(const NCSG* csg)
{
    m_csg = csg ; 
}
const NCSG* GMesh::getCSG() const 
{
    return m_csg ; 
}
const nnode* GMesh::getRoot() const 
{
    return m_csg ? m_csg->getRoot() : NULL  ; 
}

void GMesh::setCSGBoundary(const char* spec)
{
    assert( m_csg ); 

    NCSG* csg = const_cast<NCSG*>(m_csg); 
    csg->setBoundary( spec );  
}

void GMesh::setAlt(const GMesh* alt)
{
    m_alt = alt ; 
}
const GMesh* GMesh::getAlt() const 
{
    return m_alt ; 
}




void GMesh::deallocate()
{
    

    delete[] m_vertices ;  
    delete[] m_normals ;  
    delete[] m_colors ;  
    delete[] m_texcoords ;  
    delete[] m_faces ;  

    delete[] m_center_extent ;  
    delete[] m_bbox ;  
    delete[] m_transforms ;  
    delete[] m_itransforms ;  
    delete[] m_meshes ;  
    delete[] m_nodeinfo ;  
    delete[] m_identity ;  
    delete[] m_iidentity ;  

    // NB buffers and the rest are very lightweight 
}


GMesh::~GMesh()
{
    deallocate();
}

void GMesh::setVerbosity(unsigned int verbosity)
{
    m_verbosity = verbosity ; 
}

unsigned int GMesh::getVerbosity() const 
{
    return m_verbosity ; 
}

void GMesh::setName(const char* name)
{
     m_name = name ? strdup(name) : NULL ;
     if(m_name) findShortName();
}  
const char* GMesh::getName() const 
{
     return m_name ; 
}
const char* GMesh::getShortName() const 
{
     return m_shortname ; 
}


void GMesh::setVersion(const char* version)
{
     m_version = version ? strdup(version) : NULL ;
}  
const char* GMesh::getVersion() const 
{
     return m_version ; 
}




unsigned int GMesh::getIndex() const 
{
    return m_index ; 
}
unsigned int GMesh::getNumVertices() const 
{
    return m_num_vertices ; 
}
unsigned int GMesh::getNumFaces() const 
{
    return m_num_faces ; 
}
unsigned int GMesh::getNumVolumes() const 
{
    return m_num_volumes ; 
}
unsigned int GMesh::getNumVolumesSelected() const 
{
    return m_num_volumes_selected ; 
}




void GMesh::setIndex(unsigned int index)
{
   m_index = index ;
}
void GMesh::setNumVertices(unsigned int num_vertices)
{
    m_num_vertices = num_vertices ; 
}
void GMesh::setNumFaces(unsigned int num_faces)
{
    m_num_faces = num_faces ; 
}




void GMesh::setLow(gfloat3* low)
{
    m_low = low ;
}
void GMesh::setHigh(gfloat3* high)
{
    m_high = high ;
}
bool GMesh::hasTexcoords() const 
{
    return m_texcoords != NULL ;
}







gfloat3* GMesh::getLow()
{
    return m_low ;
}
gfloat3* GMesh::getHigh()
{
    return m_high ;
}
gfloat3* GMesh::getDimensions()
{
    return m_dimensions ; 
}

GMatrix<float>* GMesh::getModelToWorld()
{
    return m_model_to_world ; 
}


gfloat3* GMesh::getVertices() const 
{
    return m_vertices ;
}
gfloat3* GMesh::getNormals() const 
{
    return m_normals ;
}

gfloat3* GMesh::getColors() const 
{
    return m_colors ;
}
gfloat2* GMesh::getTexcoords() const 
{
    return m_texcoords ;
}


guint3*  GMesh::getFaces() const 
{
    return m_faces ;
}


// index is used from subclass
gfloat4 GMesh::getCenterExtent(unsigned index) const 
{
    return m_center_extent[index] ;
}

glm::vec4 GMesh::getCE(unsigned index) const 
{
    gfloat4 ce_ = getCenterExtent(index);
    glm::vec4 ce(ce_.x, ce_.y, ce_.z, ce_.w ) ; 
    return ce  ;
}

float GMesh::getBoundingRadiusCE(unsigned index) const
{
    // radius of the origin centered sphere that contains all of the volume
    // without safety margin
    glm::vec4 ce = getCE(index) ;
    float bounding_radius = glm::length(glm::vec3(ce)) + ce.w ;
    return bounding_radius ; 
}


gbbox GMesh::getBBox(unsigned int index) const 
{
    return m_bbox[index] ;
}
gbbox* GMesh::getBBoxPtr()
{
    return m_bbox ;
}






float GMesh::getExtent()
{
     return m_extent ;  
}



GBuffer*  GMesh::getModelToWorldBuffer()
{
    return (GBuffer*)m_model_to_world ;
}

float* GMesh::getModelToWorldPtr(unsigned int /*index*/)
{
     return (float*)getModelToWorldBuffer()->getPointer() ; 
}


unsigned* GMesh::getNodes() const    // CAUTION ONLY MAKES SENSE FROM GMergedMesh SUBCLASS 
{
    return m_nodes ;
}


unsigned int* GMesh::getMeshIndice() const
{
    return m_meshes ;
}
unsigned int GMesh::getMeshIndice(unsigned int index) const
{
    return m_meshes[index] ;
}




guint4* GMesh::getNodeInfo() const 
{
    return m_nodeinfo ; 
}
guint4 GMesh::getNodeInfo(unsigned int index) const
{
    return m_nodeinfo[index] ; 
}

guint4* GMesh::getIdentity() const 
{
    return m_identity ; 
}
guint4 GMesh::getIdentity(unsigned int index) const 
{
    return m_identity[index] ; 
}

guint4* GMesh::getInstancedIdentity() const
{
    return m_iidentity ; 
}
guint4 GMesh::getInstancedIdentity(unsigned int index) const
{
    return m_iidentity[index] ; 
}



unsigned int* GMesh::getBoundaries() const 
{
    return m_boundaries ;
}
unsigned int* GMesh::getSensors() const 
{
    return m_sensors ;
}




GBuffer* GMesh::getVerticesBuffer()
{
    return m_vertices_buffer ;
}
GBuffer* GMesh::getNormalsBuffer()
{
    return m_normals_buffer ;
}
GBuffer* GMesh::getColorsBuffer()
{
    return m_colors_buffer ;
}
GBuffer* GMesh::getTexcoordsBuffer()
{
    return m_texcoords_buffer ;
}
GBuffer*  GMesh::getCenterExtentBuffer()
{
    return m_center_extent_buffer ;
}
GBuffer*  GMesh::getBBoxBuffer()
{
    return m_bbox_buffer ;
}

GBuffer*  GMesh::getTransformsBuffer() const 
{
    return m_transforms_buffer ;
}
NPY<float>*  GMesh::getITransformsBuffer() const 
{
    return m_itransforms_buffer ;
}



GBuffer*  GMesh::getMeshesBuffer() const 
{
    return m_meshes_buffer ;
}
GBuffer*  GMesh::getNodeInfoBuffer() const 
{
    return m_nodeinfo_buffer ;
}
GBuffer*  GMesh::getIdentityBuffer() const 
{
    return m_identity_buffer ;
}
NPY<unsigned>*  GMesh::getInstancedIdentityBuffer() const 
{
    return m_iidentity_buffer ;
}
NPY<unsigned>*  GMesh::getComponentsBuffer() const 
{
    return m_components_buffer ;
}




GBuffer*  GMesh::getIndicesBuffer()
{
    return m_indices_buffer ;
}
GBuffer*  GMesh::getNodesBuffer()
{
    return m_nodes_buffer ;
}
GBuffer*  GMesh::getBoundariesBuffer()
{
    return m_boundaries_buffer ;
}
GBuffer*  GMesh::getSensorsBuffer()
{
    return m_sensors_buffer ;
}

bool GMesh::hasTransformsBuffer()
{
    return m_transforms_buffer != NULL ; 
}
bool GMesh::hasITransformsBuffer()
{
    return m_itransforms_buffer != NULL ; 
}







char GMesh::getGeoCode() const 
{
    return m_geocode ; 
}
void GMesh::setGeoCode(char geocode)
{
    m_geocode = geocode ; 
}


void GMesh::setInstanceSlice(NSlice* slice)
{
    m_islice = slice ; 
}
NSlice* GMesh::getInstanceSlice() const
{
    return m_islice ; 
}


void GMesh::setFaceSlice(NSlice* slice)
{
    m_fslice = slice ; 
}
NSlice* GMesh::getFaceSlice()
{
    return m_fslice ; 
}

void GMesh::setPartSlice(NSlice* slice)
{
    m_pslice = slice ; 
}
NSlice* GMesh::getPartSlice()
{
    return m_pslice ; 
}


GParts* GMesh::getParts() const 
{
    return m_parts ; 
}
void GMesh::setParts(GParts* pts) 
{
    m_parts = pts ; 
}
GPt* GMesh::getPt() const 
{
    return m_pt ; 
}
void GMesh::setPt(GPt* pt) 
{
    m_pt = pt ; 
}






void GMesh::init(gfloat3* vertices, guint3* faces, gfloat3* normals, gfloat2* texcoords)
{
   g_instance_count += 1 ; 
   setVertices(vertices);
   setFaces(faces);
   setNormals(normals);
   setTexcoords(texcoords);
   updateBounds();
   nameConstituents(m_names);
}


void GMesh::allocate()
{
    unsigned int numVertices = getNumVertices();
    unsigned int numFaces = getNumFaces();
    unsigned int numVolumes = getNumVolumes();

    bool empty = numVertices == 0 && numFaces == 0 ; 

    if(empty)
    LOG(warning) << "GMesh::allocate EMPTY"
              << " numVertices " << numVertices
              << " numFaces " << numFaces
              << " numVolumes " << numVolumes
              ;

    //assert(numVertices > 0 && numFaces > 0 && numVolumes > 0);
    assert(numVolumes > 0);

    if(numVertices > 0 && numFaces > 0)
    {

        setVertices(new gfloat3[numVertices]); 
        setNormals( new gfloat3[numVertices]);
        setColors(  new gfloat3[numVertices]);
        setTexcoords( NULL );  

        setColor(0.5,0.5,0.5);  // starting point mid-grey, change in traverse 2nd pass

        // consolidate into guint4 

        guint3* faces = new guint3[numFaces] ;
        setFacesQty(faces);
    }


    setCenterExtent(new gfloat4[numVolumes]);
    setBBox(new gbbox[numVolumes]);
    //setBBox(new nbbox[numVolumes]);
    setMeshes(new unsigned[numVolumes]);
    setNodeInfo(new guint4[numVolumes]);
    setIdentity(new guint4[numVolumes]);
    setTransforms(new float[numVolumes*16]);

    //LOG(info) << "GMesh::allocate DONE " ;
}



void GMesh::setFacesQty(guint3* faces )
{
    // TODO: consolidate into uint4 with one spare
    if(m_num_faces == 0) return ;      

    if(faces)
    {
        setFaces( faces );
    }

    setNodes(        new unsigned[m_num_faces]);
    setBoundaries(   new unsigned[m_num_faces]);
    setSensors(      new unsigned[m_num_faces]);

    for(unsigned i=0 ; i < m_num_faces ; i++)
    {
        m_nodes[i] = 0u ; 
        m_boundaries[i] = 0u ; 
        m_sensors[i] = 0u ; 
    }
}




GBuffer* GMesh::getBuffer(const char* name) const 
{
    if(isNPYBuffer(name)) return NULL ; 

    if(strcmp(name, vertices_) == 0)     return m_vertices_buffer ; 
    if(strcmp(name, normals_) == 0)      return m_normals_buffer ; 
    if(strcmp(name, colors_) == 0)       return m_colors_buffer ; 
    if(strcmp(name, texcoords_) == 0)    return m_texcoords_buffer ; 

    if(strcmp(name, indices_) == 0)      return m_indices_buffer ; 
    if(strcmp(name, nodes_) == 0)        return m_nodes_buffer ; 
    if(strcmp(name, boundaries_) == 0)   return m_boundaries_buffer ; 
    if(strcmp(name, sensors_) == 0)      return m_sensors_buffer ; 

    if(strcmp(name, center_extent_) == 0)   return m_center_extent_buffer ; 
    if(strcmp(name, bbox_) == 0)            return m_bbox_buffer ; 
    if(strcmp(name, transforms_) == 0)      return m_transforms_buffer ; 
    if(strcmp(name, meshes_) == 0)          return m_meshes_buffer ; 
    if(strcmp(name, nodeinfo_) == 0)        return m_nodeinfo_buffer ; 
    if(strcmp(name, identity_) == 0)        return m_identity_buffer ; 

    return NULL ;
}


void GMesh::setBuffer(const char* name, GBuffer* buffer)
{
    if(isNPYBuffer(name)) return ; 

    if(strcmp(name, vertices_) == 0)     setVerticesBuffer(buffer) ; 
    if(strcmp(name, normals_) == 0)      setNormalsBuffer(buffer) ; 
    if(strcmp(name, colors_) == 0)       setColorsBuffer(buffer) ; 
    if(strcmp(name, texcoords_) == 0)    setTexcoordsBuffer(buffer) ; 

    if(strcmp(name, indices_) == 0)      setIndicesBuffer(buffer) ; 
    if(strcmp(name, nodes_) == 0)        setNodesBuffer(buffer) ; 
    if(strcmp(name, boundaries_) == 0)   setBoundariesBuffer(buffer) ; 
    if(strcmp(name, sensors_) == 0)      setSensorsBuffer(buffer) ; 

    if(strcmp(name, center_extent_) == 0)   setCenterExtentBuffer(buffer) ; 
    if(strcmp(name, bbox_) == 0)            setBBoxBuffer(buffer) ; 
    if(strcmp(name, transforms_) == 0)      setTransformsBuffer(buffer) ; 
    if(strcmp(name, meshes_) == 0)          setMeshesBuffer(buffer) ; 
    if(strcmp(name, nodeinfo_) == 0)        setNodeInfoBuffer(buffer) ; 
    if(strcmp(name, identity_) == 0)        setIdentityBuffer(buffer) ; 
}



void GMesh::applyCentering()
{
    // use analytic bbox center_extent when there is an associated CSG solid

    glm::vec4 ce = m_csg ? m_csg->bbox_center_extent() : getCE(0) ; 

    LOG(debug) << " ce " << gformat(ce) ; 

    glm::vec3 tla(-ce.x, -ce.y, -ce.z); 

    applyTranslation( tla.x, tla.y, tla.z ); 

    if(m_csg)
    {
        const_cast<NCSG*>(m_csg)->set_translation( tla.x, tla.y, tla.z ); 
    }

}

void GMesh::applyTranslation(float x, float y, float z )
{
    glm::mat4 txf(nglmext::make_translate(x, y, z)); 
    GMatrixF* transform = new GMatrix<float>(glm::value_ptr(txf));
    applyTransform(*transform);  
}

void GMesh::applyTransform( GMatrixF& transform )
{
    gfloat3* vertices = getTransformedVertices(transform); 
    gfloat3* normals  = getTransformedNormals(transform); 

    unsigned num_vertices = getNumVertices() ; 

    updateVertices(vertices, num_vertices ); 
    updateNormals(normals, num_vertices ); 
    updateBounds(); 
}


void GMesh::updateVertices(gfloat3* vertices, unsigned num_vertices)
{
    assert( num_vertices == m_num_vertices ); 
 
    delete [] m_vertices ; 
    delete m_vertices_buffer ; 

    setVertices( vertices );   
}
void GMesh::setVertices(gfloat3* vertices)
{
    m_vertices = vertices ;
    m_vertices_buffer = new GBuffer( sizeof(gfloat3)*m_num_vertices, (void*)m_vertices, sizeof(gfloat3), 3 , "vertices") ;

    assert(sizeof(gfloat3) == sizeof(float)*3);
}
void GMesh::setVerticesBuffer(GBuffer* buffer)
{
    m_vertices_buffer = buffer ; 
    if(!buffer) return ; 

    m_vertices = (gfloat3*)buffer->getPointer();
    unsigned int numBytes = buffer->getNumBytes();
    m_num_vertices = numBytes/sizeof(gfloat3);
}



void GMesh::updateNormals(gfloat3* normals, unsigned num_normals)
{
    assert( num_normals == m_num_vertices ); 
 
    delete [] m_normals ; 
    delete m_normals_buffer ; 

    setNormals( normals );   
}
void GMesh::setNormals(gfloat3* normals)
{
    m_normals = normals ;
    m_normals_buffer = new GBuffer( sizeof(gfloat3)*m_num_vertices, (void*)m_normals, sizeof(gfloat3), 3 , "normals") ;
    assert(sizeof(gfloat3) == sizeof(float)*3);
}
void GMesh::setNormalsBuffer(GBuffer* buffer)
{
    m_normals_buffer = buffer ; 
    if(!buffer) return ; 
    m_normals = (gfloat3*)buffer->getPointer();
    unsigned int numBytes = buffer->getNumBytes();
    unsigned int num_normals = numBytes/sizeof(gfloat3);
    assert( m_num_vertices == num_normals );  // must load vertices before normals
}

void GMesh::setColors(gfloat3* colors)
{
    m_colors = colors ;
    m_colors_buffer = new GBuffer( sizeof(gfloat3)*m_num_vertices, (void*)m_colors, sizeof(gfloat3), 3 , "colors") ;
    assert(sizeof(gfloat3) == sizeof(float)*3);
}
void GMesh::setColorsBuffer(GBuffer* buffer)
{
    m_colors_buffer = buffer ; 
    if(!buffer) return ; 
    
    m_colors = (gfloat3*)buffer->getPointer();
    unsigned int numBytes = buffer->getNumBytes();
    unsigned int num_colors = numBytes/sizeof(gfloat3);

    assert( m_num_vertices == num_colors );  // must load vertices before colors
}


void GMesh::setCenterExtent(gfloat4* center_extent)  
{
    m_center_extent = center_extent ;  

    LOG(debug) << "GMesh::setCenterExtent (creates buffer) " 
              << " m_center_extent " << m_center_extent
              << " m_num_volumes " << m_num_volumes 
              ; 

    assert(m_num_volumes > 0);
    m_center_extent_buffer = new GBuffer( sizeof(gfloat4)*m_num_volumes, (void*)m_center_extent, sizeof(gfloat4), 4 , "cen_ext"); 
    assert(sizeof(gfloat4) == sizeof(float)*4);
}
void GMesh::setCenterExtentBuffer(GBuffer* buffer) 
{
    m_center_extent_buffer = buffer ;  
    if(!buffer) return ; 

    m_center_extent = (gfloat4*)buffer->getPointer();
    unsigned int numBytes = buffer->getNumBytes();
    m_num_volumes = numBytes/sizeof(gfloat4) ;

    LOG(debug) << "GMesh::setCenterExtentBuffer  (creates array from buffer) " 
              << " m_center_extent " << m_center_extent
              << " m_num_volumes " << m_num_volumes 
              ; 



}


void GMesh::setBBox(gbbox* bb)  
{
    m_bbox = bb ;  
    assert(m_num_volumes > 0);
    m_bbox_buffer = new GBuffer( sizeof(gbbox)*m_num_volumes, (void*)m_bbox, sizeof(gbbox), 6 , "bbox"); 
    assert(sizeof(gbbox) == sizeof(float)*6);
}
void GMesh::setBBoxBuffer(GBuffer* buffer) 
{
    m_bbox_buffer = buffer ;  
    if(!buffer) return ; 

    m_bbox = (gbbox*)buffer->getPointer();
    unsigned int numBytes = buffer->getNumBytes();
    unsigned int numVolumes = numBytes/sizeof(gbbox) ;

    setNumVolumes(numVolumes);
}



void GMesh::setTransforms(float* transforms)  
{
    m_transforms = transforms ;  
    assert(m_num_volumes > 0);

    unsigned int numElements = 16 ; 
    unsigned int size = sizeof(float)*numElements;

    LOG(debug) << "GMesh::setTransforms " 
              << " num_volumes " << m_num_volumes 
              << " size " << size 
              << " fsize " << sizeof(float)
              ;

    m_transforms_buffer = new GBuffer( size*m_num_volumes, (void*)m_transforms, size, numElements , "transforms"); 
}


void GMesh::setTransformsBuffer(GBuffer* buffer) 
{
    m_transforms_buffer = buffer ;  
    if(!buffer) return ; 
    m_transforms = (float*)buffer->getPointer();
}

void GMesh::setITransformsBuffer(NPY<float>* buffer) 
{
    m_itransforms_buffer = buffer ;  
    if(!buffer) return ; 
    m_itransforms = buffer->getValues();
}






unsigned int GMesh::getNumTransforms() const 
{
    return m_transforms_buffer ? m_transforms_buffer->getNumBytes()/(16*sizeof(float)) : 0 ; 
}
unsigned int GMesh::getNumITransforms() const 
{
    if(!m_itransforms_buffer) return 0 ;    
    unsigned int n0 = m_itransforms_buffer->getNumBytes()/(16*sizeof(float)) ; 
    unsigned int n1 = m_itransforms_buffer->getNumItems() ;
    assert(n0 == n1); 
    return n1 ;  
}


float* GMesh::getTransform(unsigned int index)
{
    if(index >= m_num_volumes)
    {
        LOG(fatal) << "GMesh::getTransform out of bounds " 
                     << " m_num_volumes " << m_num_volumes 
                     << " index " << index
                     ;
        assert(0);
    }
    return index < m_num_volumes ? m_transforms + index*16 : NULL  ;
}

float* GMesh::getITransform(unsigned int index)
{
    unsigned int num_itransforms = getNumITransforms();
    return index < num_itransforms ? m_itransforms + index*16 : NULL  ;
}










void GMesh::setMeshes(unsigned int* meshes)  
{
    m_meshes = meshes ;  
    assert(m_num_volumes > 0);
    unsigned int size = sizeof(unsigned int);
    m_meshes_buffer = new GBuffer( size*m_num_volumes, (void*)m_meshes, size, 1 , "meshes"); 
}

void GMesh::setMeshesBuffer(GBuffer* buffer) 
{
    m_meshes_buffer = buffer ;  
    if(!buffer) return ; 

    m_meshes = (unsigned int*)buffer->getPointer();
    unsigned int numBytes = buffer->getNumBytes();
    unsigned int numElements = 1  ; 
    unsigned int size = sizeof(float)*numElements;
    unsigned int numVolumes = numBytes/size ;
    setNumVolumes(numVolumes);
}



void GMesh::setNodeInfo(guint4* nodeinfo)  
{
    m_nodeinfo = nodeinfo ;  
    assert(m_num_volumes > 0);
    unsigned int size = sizeof(guint4);
    assert(size == sizeof(unsigned int)*4 );
    m_nodeinfo_buffer = new GBuffer( size*m_num_volumes, (void*)m_nodeinfo, size, 4 , "nodeinfo"); 
}
void GMesh::setNodeInfoBuffer(GBuffer* buffer) 
{
    m_nodeinfo_buffer = buffer ;  
    if(!buffer) return ; 

    m_nodeinfo = (guint4*)buffer->getPointer();
    unsigned int numBytes = buffer->getNumBytes();
    unsigned int size = sizeof(guint4);
    assert(size == sizeof(unsigned int)*4 );
    unsigned int numVolumes = numBytes/size ;
    setNumVolumes(numVolumes);
}




void GMesh::setIdentity(guint4* identity)  
{
    m_identity = identity ;  
    assert(m_num_volumes > 0);
    unsigned int size = sizeof(guint4);
    assert(size == sizeof(unsigned int)*4 );
    m_identity_buffer = new GBuffer( size*m_num_volumes, (void*)m_identity, size, 4 , "identity"); 
}
void GMesh::setIdentityBuffer(GBuffer* buffer) 
{
    m_identity_buffer = buffer ;  
    if(!buffer) return ; 

    m_identity = (guint4*)buffer->getPointer();
    unsigned int numBytes = buffer->getNumBytes();
    unsigned int size = sizeof(guint4);
    assert(size == sizeof(unsigned int)*4 );
    unsigned int numVolumes = numBytes/size ;
    setNumVolumes(numVolumes);
}




void GMesh::setInstancedIdentityBuffer(NPY<unsigned int>* buffer) 
{
    m_iidentity_buffer = buffer ;  
    if(!buffer) return ; 
    m_iidentity = (guint4*)buffer->getPointer();
}






void GMesh::setComponentsBuffer(NPY<unsigned>* buf)
{
    m_components_buffer = buf ;
}

int GMesh::getNumComponents() const 
{
    return m_components_buffer ? m_components_buffer->getShape(0) : -1 ; 
}

void GMesh::getComponent( glm::uvec4& eidx, unsigned icomp ) const 
{
     eidx = m_components_buffer->getQuad(icomp);
}


void GMesh::setComponent(const glm::uvec4& eidx, unsigned icomp )
{
    assert( m_num_mergedmesh > 0 && "MUST GMergedMesh::countMergedMesh before GMesh::setComponent ");
    if(!m_components_buffer) 
    {
         NPY<unsigned>* comp = NPY<unsigned>::make(m_num_mergedmesh, 4);
         comp->zero();
         setComponentsBuffer(comp); 
    }
    assert( icomp < m_num_mergedmesh );
    m_components_buffer->setQuad(eidx, icomp );
}
 
void GMesh::dumpComponents(const char* msg) const 
{
    LOG(info) << msg 
              << " numComponents " << getNumComponents() 
               ;

    if(getNumComponents() < 1 ) return ;
    unsigned num_comp = getNumComponents();
    for(unsigned icomp=0 ; icomp < num_comp ; icomp++)
    {
         glm::uvec4 eidx ;
         getComponent( eidx, icomp );
         std::cout << std::setw(4) << icomp
                   << gpresent(eidx)
                   << std::endl 
                   ;

    }

}








void GMesh::setNumVolumes(unsigned int numVolumes)
{
    if(m_num_volumes == 0) 
    {
        m_num_volumes = numVolumes ; 
    }
    else
    {
        assert(numVolumes == m_num_volumes);
    }
}


void GMesh::setTexcoords(gfloat2* texcoords)
{
    if(!texcoords) return ;
    m_texcoords = texcoords ;
    m_texcoords_buffer = new GBuffer( sizeof(gfloat2)*m_num_vertices, (void*)m_texcoords, sizeof(gfloat2), 2 , "texcoords") ;
    assert(sizeof(gfloat2) == sizeof(float)*2);
}
void GMesh::setTexcoordsBuffer(GBuffer* buffer)
{
    m_texcoords_buffer = buffer ; 
    if(!buffer) return ; 

    m_texcoords = (gfloat2*)buffer->getPointer(); 
    unsigned int numBytes = buffer->getNumBytes();
    unsigned int num_texcoords = numBytes/sizeof(gfloat2);
    assert( m_num_vertices == num_texcoords );  // must load vertices before texcoords
}


void GMesh::setFaces(guint3* faces)
{
    assert(sizeof(guint3) == 3*4);
    unsigned int totbytes = sizeof(guint3)*m_num_faces ;
    unsigned int itemsize = sizeof(guint3)/3 ;           // this regards the item as the individual index integer
    unsigned int nelem    = 1 ;                          // number of elements in the item

    m_faces = faces ;
    m_indices_buffer = new GBuffer( totbytes, (void*)m_faces, itemsize, nelem , "indices") ;
    assert(sizeof(guint3) == sizeof(unsigned int)*3);
}
void GMesh::setIndicesBuffer(GBuffer* buffer)
{
    m_indices_buffer = buffer ; 
    if(!buffer) return ;

    m_faces = (guint3*)buffer->getPointer();
    unsigned int numBytes = buffer->getNumBytes();
    m_num_faces = numBytes/sizeof(guint3);    // NB kludge equating "int" buffer to "unsigned int" 
}


void GMesh::setNodes(unsigned* nodes)   // only makes sense to use from single subclasses instances like GMergedMesh 
{
    m_nodes = nodes ;
    m_nodes_buffer = new GBuffer( sizeof(unsigned int)*m_num_faces, (void*)m_nodes, sizeof(unsigned int), 1 , "nodes") ;
    assert(sizeof(unsigned int) == sizeof(unsigned int)*1);
}
void GMesh::setNodesBuffer(GBuffer* buffer)
{
    m_nodes_buffer = buffer ; 
    if(!buffer) return ;

    m_nodes = (unsigned int*)buffer->getPointer();
    unsigned int numBytes = buffer->getNumBytes();
    unsigned int num_nodes = numBytes/sizeof(unsigned int);

    // assert(m_num_faces == num_nodes);   // must load indices before nodes
    if(m_num_faces != num_nodes)
        LOG(warning) << "GMesh::setNodesBuffer allowing inconsistency " ; 
}


void GMesh::setBoundaries(unsigned* boundaries)
{
    m_boundaries = boundaries ;
    m_boundaries_buffer = new GBuffer( sizeof(unsigned)*m_num_faces, (void*)m_boundaries, sizeof(unsigned), 1 , "boundaries") ;
    assert(sizeof(unsigned) == sizeof(unsigned)*1);
}
void GMesh::setBoundariesBuffer(GBuffer* buffer)
{
    m_boundaries_buffer = buffer ; 
    if(!buffer) return ;

    m_boundaries = (unsigned int*)buffer->getPointer();

    unsigned int numBytes = buffer->getNumBytes();
    unsigned int num_boundaries = numBytes/sizeof(unsigned int);

    // assert(m_num_faces == num_boundaries);   // must load indices before boundaries, for m_num_faces
    if(m_num_faces != num_boundaries)
        LOG(warning) << "GMesh::setBoundariesBuffer allowing inconsistency " ; 
}



void GMesh::setSensors(unsigned* sensors)
{
    m_sensors = sensors ;
    m_sensors_buffer = new GBuffer( sizeof(unsigned)*m_num_faces, (void*)m_sensors, sizeof(unsigned), 1 , "sensors") ;
    assert(sizeof(unsigned) == sizeof(unsigned)*1);
}
void GMesh::setSensorsBuffer(GBuffer* buffer)
{
    m_sensors_buffer = buffer ; 
    if(!buffer) return ;

    m_sensors = (unsigned*)buffer->getPointer();

    unsigned numBytes = buffer->getNumBytes();
    unsigned num_sensors = numBytes/sizeof(unsigned);
    assert(m_num_faces == num_sensors);   // must load indices before sensors, for m_num_faces
}




void GMesh::setColor(float r, float g, float b)
{
    //assert(m_num_colors == m_num_vertices);
    if(!m_colors)
    {
        setColors(new gfloat3[m_num_vertices]);
    }
    for(unsigned int i=0 ; i<m_num_vertices ; ++i )
    {
        m_colors[i].x  = r ;
        m_colors[i].y  = g ;
        m_colors[i].z  = b ;
    }
}

void GMesh::dumpNormals(const char* msg, unsigned int nmax) const 
{
    LOG(info) << msg  ;
    LOG(info) << " num_vertices " << m_num_vertices 
              ;  

    for(unsigned int i=0 ; i < std::min(nmax,m_num_vertices) ; i++)
    {
        gfloat3& nrm = m_normals[i] ;
        printf(" nrm %5u  %10.3f %10.3f %10.3f \n", i, nrm.x, nrm.y, nrm.z );
    } 
}

void GMesh::dump(const char* msg, unsigned int nmax) const 
{
    LOG(info) << msg  
              << " num_vertices " << m_num_vertices 
              << " num_faces " << m_num_faces
              << " num_volumes " << m_num_volumes
              << " name " << ( m_name ? m_name : "-" )
              ;  

    std::cout << " low  " << (m_low ?  m_low->desc() : "-" ) << std::endl ; 
    std::cout << " high " << (m_high ? m_high->desc() : "-" ) << std::endl ; 
    std::cout << " dim  " << (m_dimensions ? m_dimensions->desc() : "-" ) << std::endl ; 
    std::cout << " cen  " << (m_center ? m_center->desc() : "-" ) << " extent " << m_extent << std::endl ; 
    std::cout << " ce   " << (m_center_extent ? m_center_extent->desc() : "-" )  << std::endl ; 

    if(m_bbox)
    {
        std::cout << " bb.max   " << m_bbox->max.desc()  << std::endl ; 
        std::cout << " bb.min   " << m_bbox->min.desc()  << std::endl ; 
    }


    for(unsigned int i=0 ; i < std::min(nmax,m_num_vertices) ; i++)
    {
        gfloat3& vtx = m_vertices[i] ;
        gfloat3& nrm = m_normals[i] ;
        std::cout << std::setw(5) << i 
                  << " vtx " << vtx.desc() 
                  << " nrm " << nrm.desc() 
                  << std::endl ; 

    } 
    std::cout << std::endl ;   

    if(hasTexcoords())
    { 
        for(unsigned int i=0 ; i < std::min(nmax,m_num_vertices) ; i++)
        {
            gfloat2& tex = m_texcoords[i] ;
            printf(" tex %5u  %10.3f %10.3f  \n", i, tex.u, tex.v );
        } 
    }


    if(m_colors)
    {
        for(unsigned int i=0 ; i < std::min(nmax,m_num_vertices) ; i++)
        {
            gfloat3& col = m_colors[i] ;
            printf(" col %5u  %10.3f %10.3f %10.3f \n", i, col.x, col.y, col.z );
        }
    } 


    LOG(info) << " num_faces " << m_num_faces ;  

    for(unsigned int i=0 ; i < std::min(nmax,m_num_faces) ; i++)
    {
        guint3& fac = m_faces[i] ;
        printf(" fac %5u  %5u %5u %5u \n", i, fac.x, fac.y, fac.z );
    } 

    if(m_nodes && m_boundaries)
    {
        for(unsigned int i=0 ; i < std::min(nmax,m_num_faces) ; i++)
        {
            unsigned int& node = m_nodes[i] ;
            unsigned int& boundary = m_boundaries[i] ;
            printf(" fac %5u  node %5u boundary %5u  \n", i, node, boundary );
        } 
    }

}




std::string GMesh::desc() const  
{
    std::stringstream ss ; 
    unsigned nv = getNumVertices();
    unsigned nf = getNumFaces();
    //unsigned nc = getNumColors();
    ss 
        << " nv " << std::setw(6) << nv 
        << " nf " << std::setw(6) << nf
      //  << " nc " << std::setw(6) << nc
        ;

    return ss.str();
}

void GMesh::Summary(const char* msg) const 
{
   LOG(info) << msg ;  

   printf("%s idx %u vx %u fc %u n %s sn %s \n",
      msg, 
      m_index, 
      m_num_vertices, 
      m_num_faces,
      m_name,
      m_shortname
   );

   if(m_low) printf("%10s %10.3f %10.3f %10.3f\n",
         "low",
         m_low->x,
         m_low->y,
         m_low->z);

   if(m_high) printf("%10s %10.3f %10.3f %10.3f\n", 
          "high",
          m_high->x,
          m_high->y,
          m_high->z);

   if(m_dimensions) printf("%10s %10.3f %10.3f %10.3f extent %10.3f\n", 
          "dimen",
          m_dimensions->x,
          m_dimensions->y,
          m_dimensions->z,
          m_extent);

   if(m_center) printf("%10s %10.3f %10.3f %10.3f\n", 
          "center",
          m_center->x,
          m_center->y,
          m_center->z);

   if(m_center_extent) printf("%10s %10.3f %10.3f %10.3f %10.3f \n", 
          "center_extent",
          m_center_extent->x,
          m_center_extent->y,
          m_center_extent->z,
          m_center_extent->w);

   //m_model_to_world->Summary(msg);
}



gbbox* GMesh::findBBox(gfloat3* vertices, unsigned int num_vertices)
{
    if(num_vertices == 0) return NULL ;

    std::vector<glm::vec3> points ; 

    for( unsigned int i = 0; i < num_vertices ;++i )
    {
        gfloat3& v = vertices[i];
        glm::vec3 p(v.x,v.y,v.z);
        points.push_back(p);
    }

    unsigned verbosity = 0 ;  
    nbbox nbb = nbbox::from_points(points, verbosity);
    gbbox* bb = new gbbox(nbb);

    return bb ; 
} 


gfloat4 GMesh::findCenterExtentDeprecated(gfloat3* vertices, unsigned int num_vertices)
{
    gfloat3  low( 1e10f, 1e10f, 1e10f);
    gfloat3 high( -1e10f, -1e10f, -1e10f);

    for( unsigned int i = 0; i < num_vertices ;++i )
    {
        gfloat3& v = vertices[i];

        low.x = std::min( low.x, v.x);
        low.y = std::min( low.y, v.y);
        low.z = std::min( low.z, v.z);

        high.x = std::max( high.x, v.x);
        high.y = std::max( high.y, v.y);
        high.z = std::max( high.z, v.z);
    }

    gfloat3 dimensions(high.x - low.x, high.y - low.y, high.z - low.z );
    float extent = 0.f ;
    extent = std::max( dimensions.x , extent );
    extent = std::max( dimensions.y , extent );
    extent = std::max( dimensions.z , extent );
    extent = extent / 2.0f ;         
 
    gfloat4 center_extent((high.x + low.x)/2.0f, (high.y + low.y)/2.0f , (high.z + low.z)/2.0f, extent );

    return center_extent ; 
}


/**
GMesh::updatebounds
--------------------

Updates the below based on the bounding box of the vertices.
The vertices present depend on the geometry selection 
when the GMergedMesh was created from the GMesh of the GVolume. 

::

    m_center_extent[0] 
    m_bbox[0]
    m_model_to_world


updateBounds is invoked from::

   GMesh::init
   GMesh::loadBuffers
   GGeo::invokeMeshJoin
   GMergedMesh::combine
   GMergedMesh::create
 
TODO-SOMETIME: 

Avoid the need for updating as it is a source of confusion
arising from double use of slot zero for:

1. absolute volume 0 (world volume, an often over large container)  
2. overall selected volumes 


model coordinates definition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* all vertices are contained within model coordinates box  (-1:1,-1:1,-1:1) 
* model coordinates origin (0,0,0) corresponds to world coordinates  m_center
* extent is half the maximal dimension 

world -> model
   
* translate by -m_center 
* scale by 1/m_extent

model -> world

* scale by m_extent
* translate by m_center


TO FIX 
~~~~~~~~~~~~

This avoid stomping on position of array of center_extent in case of MergedMesh, 
but it still overwrites volume 0 


**/

void GMesh::updateBounds()
{
    gbbox*  bb = findBBox(m_vertices, m_num_vertices);

    gfloat4 ce(0,0,0,1.f) ;
    if(bb) ce = bb->center_extent() ;

    LOG(debug) << " ce " <<  ce.desc(); 

    m_model_to_world = new GMatrix<float>( ce.x, ce.y, ce.z, ce.w );

    if(bb)
    {
        if(m_bbox == NULL)
        {
            m_bbox = new gbbox(*bb) ;
        }
        else
        {
            m_bbox[0].min = bb->min ;
            m_bbox[0].max = bb->max ;
        }
    }

    if(m_center_extent == NULL)
    {
        m_center_extent = new gfloat4( ce.x, ce.y, ce.z, ce.w );
    }
    else
    {
        LOG(debug)
            << "(to bbox of selected vertices) "
            << " overwrite volume 0 ce "
            <<  m_center_extent[0].description()
            << " with " 
            << ce.description()
            << " num_vertices " << m_num_vertices
            << " num_volumes " << m_num_volumes
            ;

        m_center_extent[0].x = ce.x ;
        m_center_extent[0].y = ce.y ;
        m_center_extent[0].z = ce.z ;
        m_center_extent[0].w = ce.w ;
    }
}



void GMesh::updateBounds(gfloat3& low, gfloat3& high, GMatrixF& transform)
{
    if(m_low && m_high)
    {   
        gfloat3 mlow(*m_low) ; 
        gfloat3 mhigh(*m_high) ; 

        mlow  *= transform ; 
        mhigh *= transform ; 

        low.x = std::min( low.x, mlow.x);
        low.y = std::min( low.y, mlow.y);
        low.z = std::min( low.z, mlow.z);

        high.x = std::max( high.x, mhigh.x);
        high.y = std::max( high.y, mhigh.y);
        high.z = std::max( high.z, mhigh.z);
   }
}







gfloat3* GMesh::getTransformedVertices(GMatrixF& transform ) const 
{
     gfloat3* vertices = new gfloat3[m_num_vertices];
     for(unsigned int i = 0; i < m_num_vertices; i++)
     {  
         vertices[i].x = m_vertices[i].x ;   
         vertices[i].y = m_vertices[i].y ;   
         vertices[i].z = m_vertices[i].z ;   

         vertices[i] *= transform ;
     }   
     return vertices ;
}

gfloat3* GMesh::getTransformedNormals(GMatrixF& transform ) const 
{
     gfloat3* normals = new gfloat3[m_num_vertices];
     for(unsigned int i = 0; i < m_num_vertices; i++)
     {  
         gfloat4 nrm(m_normals[i], 0.);   // w=0 as direction, not position 

         nrm *= transform ; 
         // NB should be transpose of inverse, 
         // (so only OK if orthogonal, that means rotations only no non-uniform scaling)

         normals[i].x = nrm.x ; 
         normals[i].y = nrm.y ; 
         normals[i].z = nrm.z ; 

     }   
     return normals ;
}



void GMesh::updateDistinctBoundaries()
{
    for(unsigned int i=0 ; i < getNumFaces() ; i++)
    {
        unsigned int index = m_boundaries[i] ;
        if(std::count(m_distinct_boundaries.begin(), m_distinct_boundaries.end(), index ) == 0) m_distinct_boundaries.push_back(index);
    }  
    std::sort( m_distinct_boundaries.begin(), m_distinct_boundaries.end() );
}
 
std::vector<unsigned int>& GMesh::getDistinctBoundaries()
{
    if(m_distinct_boundaries.size()==0) updateDistinctBoundaries();
    return m_distinct_boundaries ;
}







bool GMesh::isFloatBuffer(const char* name) const 
{

    return ( strcmp( name, vertices_) == 0 || 
             strcmp( name, normals_) == 0  || 
             strcmp( name, center_extent_) == 0  || 
             strcmp( name, bbox_) == 0  || 
             strcmp( name, transforms_) == 0  || 
             strcmp( name, colors_) == 0 );
}

bool GMesh::isIntBuffer(const char* name) const 
{ 
    return ( 
             strcmp( name, indices_) == 0     || 
             strcmp( name, nodes_) == 0       || 
             strcmp( name, sensors_) == 0     || 
             strcmp( name, boundaries_) == 0 
          );
}
bool GMesh::isUIntBuffer(const char* name) const 
{
    return 
           ( 
              strcmp( name, nodeinfo_) == 0  ||
              strcmp( name, identity_) == 0  ||
              strcmp( name, iidentity_) == 0  ||
              strcmp( name, meshes_) == 0  
           );
}


bool GMesh::isNPYBuffer(const char* name)  const 
{
    return 
           ( 
              strcmp( name, iidentity_) == 0  ||
              strcmp( name, itransforms_) == 0  ||
              strcmp( name, components_) == 0  
           );
}


void GMesh::saveBuffer(const char* path, const char* name, GBuffer* buffer) const 
{
    LOG(debug) << "GMesh::saveBuffer "
               << " name " << std::setw(25) << name 
               << " path " << path  
               ;

    if(isNPYBuffer(name))  
    {
         saveNPYBuffer(path, name);
    }
    else if(buffer != NULL)
    {
        if(isFloatBuffer(name))     buffer->save<float>(path);
        else if(isIntBuffer(name))  buffer->save<int>(path);
        else if(isUIntBuffer(name)) buffer->save<unsigned>(path);
        else 
           printf("GMesh::saveBuffer WARNING NOT saving uncharacterized buffer %s into %s \n", name, path );
    }
}

void GMesh::saveNPYBuffer(const char* path, const char* name) const 
{
    NPYBase* buf = getNPYBuffer(name);
    if(buf)
    {
        buf->save(path);
    }
    else
    {
        LOG(debug) << "GMesh::saveNPYBuffer"
                     << " NULL buffer not saving "
                     << " path " << path
                     << " name " << name 
        ;
    } 
}

NPYBase* GMesh::getNPYBuffer(const char* name) const 
{
    NPYBase* buf(NULL);
    if(strcmp(name, iidentity_) == 0)   buf = getInstancedIdentityBuffer();
    else if(strcmp(name, itransforms_) == 0) buf = getITransformsBuffer();
    else if(strcmp(name, components_) == 0)  buf = getComponentsBuffer();
    return buf ; 
}


void GMesh::loadNPYBuffer(const char* path, const char* name)
{
    LOG(debug) << "GMesh::loadNPYBuffer" 
              << " name " << name
              << " path " << path 
              ; 

    if(strcmp(name, iidentity_) == 0)
    {
        NPY<unsigned>* buf = NPY<unsigned>::load(path) ;
        setInstancedIdentityBuffer(buf);
    }
    else if(strcmp(name, itransforms_) == 0)
    {
        NPY<float>* buf = NPY<float>::load(path) ;
        setITransformsBuffer(buf);
    }
    else if(strcmp(name, components_) == 0)
    {
        NPY<unsigned>* buf = NPY<unsigned>::load(path) ;
        setComponentsBuffer(buf);
    }
    else
    {
        assert(0);
    }
}


void GMesh::loadBuffer(const char* path, const char* name)
{
    if(isNPYBuffer(name))
    {
         loadNPYBuffer(path, name);
    }
    else
    {
        GBuffer* buffer(NULL); 
        if(isFloatBuffer(name))                    buffer = GBuffer::load<float>(path);
        else if(isIntBuffer(name))                 buffer = GBuffer::load<int>(path);
        else if(isUIntBuffer(name))                buffer = GBuffer::load<unsigned int>(path);
        else
            printf("GMesh::loadBuffer WARNING not loading %s from %s \n", name, path ); 

        if(buffer) setBuffer(name, buffer);
    }
}



std::vector<std::string>& GMesh::getNames()
{
    return m_names ; 
}

std::string GMesh::getVersionedBufferName(std::string& name) const 
{
    std::string vname = name ;
    if(m_version)
    {
        if(vname.compare("vertices") == 0 || 
           vname.compare("indices") == 0  || 
           vname.compare("colors") == 0  || 
           vname.compare("normals") == 0)
           { 
               vname += m_version ;
               LOG(warning) << "GMesh::loadBuffers version setting changed buffer name to " << vname ; 
           }
    }
    return vname ; 
}



GMesh* GMesh::load(const char* dir, const char* typedir, const char* instancedir)
{

    std::string cachedir = BFile::FormPath(dir, typedir, instancedir);
    bool existsdir = BFile::ExistsDir(dir, typedir, instancedir);

    LOG(debug) << "GMesh::load"
              << " dir " << dir 
              << " typedir " << typedir 
              << " instancedir " << instancedir 
              << " cachedir " << cachedir 
              << " existsdir " << existsdir
              ;
 


    GMesh* mesh(NULL);
    if(!existsdir)
    {
        LOG(error)  << "GMesh::load FAILED : NO DIRECTORY "
                    << " dir " << dir
                    << " typedir " << typedir
                    << " instancedir " << instancedir
                    << " -> cachedir " << cachedir
                    ;
    }
    else
    {
        mesh = new GMesh(0, NULL, 0, NULL, 0, NULL, NULL );
        mesh->loadBuffers(cachedir.c_str());
    }
    return mesh ; 
}


void GMesh::save(const char* dir, const char* typedir, const char* instancedir) const 
{
    std::string cachedir = BFile::CreateDir(dir, typedir, instancedir);

    if(!cachedir.empty())
     {
        const char* dir = cachedir.c_str() ;   
        saveBuffers(dir);
    }
    else 
    {
        LOG(error)  << "GMesh::save FAILED : NO DIRECTORY "
                    << " dir " << dir
                    << " typedir " << typedir
                    << " instancedir " << instancedir
                    << " -> cachedir " << cachedir 
                    ;
    }
}


void GMesh::loadBuffers(const char* dir)
{
    LOG(verbose) << "GMesh::loadBuffers " << dir ;  

    for(unsigned int i=0 ; i<m_names.size() ; i++)
    {
        std::string name = m_names[i];
        std::string vname = getVersionedBufferName(name);  
        fs::path bufpath(dir);
        bufpath /= vname + ".npy" ; 

        if(fs::exists(bufpath) && fs::is_regular_file(bufpath))
        { 
            loadBuffer(bufpath.string().c_str(), name.c_str());
        }
        else
        {
            LOG(verbose) << "no such bufpath: " << bufpath ; 
        }
    } 
    updateBounds();
}


void GMesh::saveBuffers(const char* dir) const 
{
    for(unsigned int i=0 ; i<m_names.size() ; i++)
    {
        std::string name = m_names[i];
        std::string vname = getVersionedBufferName(name);  
        fs::path bufpath(dir);
        bufpath /= vname + ".npy" ; 

        GBuffer* buffer = getBuffer(name.c_str());

        saveBuffer(bufpath.string().c_str(), name.c_str(), buffer);  
    } 
}


GMesh* GMesh::makeDedupedCopy()
{
    GMeshFixer fixer(this);
    fixer.copyWithoutVertexDuplicates();   
    return fixer.getDst(); 
}


GMesh* GMesh::load_deduped(const char* dir, const char* typedir, const char* instancedir)
{
    LOG(verbose) << "GMesh::load_deduped"
               << " dir " << dir
               << " typedir " << typedir
               << " instancedir " << instancedir
               ;

    GMesh* gm = GMesh::load(dir, typedir, instancedir) ;

    if(!gm)
    {
         LOG(error) << "GMesh::load_deduped FAILED to load mesh"
               << " dir " << dir
               << " typedir " << typedir
               << " instancedir " << instancedir
               ;

         return NULL ; 
    }    

    GMesh* dm = gm->makeDedupedCopy();
    delete gm ; 
    return dm ; 
}

void GMesh::findShortName()
{
   if(!m_name) return ; 
   m_shortname = BStr::trimPointerSuffixPrefix(m_name, NULL );   
}


/**
GMesh::makeFaceRepeatedInstancedIdentityBuffer
-----------------------------------------------------

Canonically invoked by optixrap-/OGeo::makeTriangulatedGeometry
Constructing a face repeated IIdentity buffer
to be addressed with 0:numInstances*PrimitiveCount::

   instanceIdx*PrimitiveCount + primIdx ;

where the primIdx goes over all the volumes 

**/

GBuffer* GMesh::makeFaceRepeatedInstancedIdentityBuffer()
{
    unsigned int numITransforms = getNumITransforms() ;
    if(numITransforms == 0)
    {
        LOG(warning) << "GMesh::makeFaceRepeatedInstancedIdentityBuffer only relevant to instanced meshes " 
                     << " m_index " <<  m_index
                     << " numITransforms " << numITransforms
                     ;
        return NULL ; 
    }


    unsigned numVolumes = getNumVolumes();
    unsigned numVolumesSelected = getNumVolumesSelected();
    unsigned numFaces = getNumFaces() ;
    unsigned numRepeatedIdentity = numITransforms*numFaces ;
    unsigned numInstanceIdentity = m_iidentity_buffer->getShape(0)*m_iidentity_buffer->getShape(1) ;  

    LOG(LEVEL)
        << " m_index " << m_index
        << " numITransforms " << numITransforms
        << " numVolumes " << numVolumes
        << " numVolumesSelected " << numVolumesSelected
        << " numFaces " << numFaces
        << " numRepeatedIdentity (numITransforms*numFaces) " << numRepeatedIdentity
        << " numInstanceIdentity " << numInstanceIdentity
        << " m_iidentity_buffer " << m_iidentity_buffer->getShapeString()
        << " m_itransforms_buffer " << m_itransforms_buffer->getShapeString()
        ;


    bool nodeinfo_ok = m_nodeinfo_buffer && m_nodeinfo_buffer->getNumItems() == numVolumes ;
    bool iidentity_ok = m_iidentity_buffer && numInstanceIdentity == numVolumes*numITransforms ;


    if(!nodeinfo_ok)
        LOG(fatal) 
            << " nodeinfo_ok " << nodeinfo_ok
            << " nodeinfo_buffer_items " << ( m_nodeinfo_buffer ? m_nodeinfo_buffer->getNumItems() : -1 )
            << " numVolumes " << numVolumes  
            ;

    if(!iidentity_ok)
       LOG(fatal) 
           << " iidentity_ok " << iidentity_ok
           << " iidentity_buffer_items " << ( m_iidentity_buffer ? m_iidentity_buffer->getNumItems() : -1 )
           << " numFaces (sum of faces in numVolumes)" << numFaces 
           << " numITransforms " << numITransforms
           << " numVolumes*numITransforms " << numVolumes*numITransforms 
           << " numRepeatedIdentity " << numRepeatedIdentity 
           ; 

    assert(nodeinfo_ok);
    assert(iidentity_ok);

    guint4* nodeinfo = getNodeInfo();
    unsigned int nftot(0);
    unsigned int offset(0);
    unsigned int i1 = numFaces ;                    // instance 1 offset
    unsigned int il = (numITransforms-1)*numFaces ; // instance N-1 offset 

    // check nodeinfo per-volume sum of faces matches expected total 

    if(m_verbosity > 3) 
    LOG(info) << "GMesh::makeFaceRepeatedInstancedIdentityBuffer"
              << " verbosity " << m_verbosity
              << " dumping per volume offsets "
              ;

    for(unsigned s=0 ; s < numVolumes ; s++)
    {
        unsigned nf = (nodeinfo + s)->x ;
        if(m_verbosity > 3)
        printf(" s %u nf %3d  i0 %d:%d  i1 %d:%d   il %d:%d \n", s, nf, offset, offset+nf, i1+offset, i1+offset+nf, il+offset, il+offset+nf ); 
        nftot += nf ;
        offset += nf ; 
    }

    if(m_verbosity > 3) 
    LOG(info) << "GMesh::makeFaceRepeatedInstancedIdentityBuffer"
              << " verbosity " << m_verbosity
              << " nftot " << nftot
              ;
    assert( numFaces == nftot );




    guint4* riid = new guint4[numRepeatedIdentity] ;
    
    for(unsigned i=0 ; i < numITransforms ; i++)
    {
        offset = 0 ; 
        for(unsigned s=0 ; s < numVolumes ; s++)
        {   
            guint4 iid = m_iidentity[numVolumes*i + s]  ;  
            unsigned nf = (nodeinfo + s)->x ;
            for(unsigned f=0 ; f < nf ; ++f) riid[i*numFaces+offset+f] = iid ; 
            offset += nf ; 
        }  
    }   
    
    unsigned int size = sizeof(guint4) ;
    GBuffer* buffer = new GBuffer( size*numRepeatedIdentity, (void*)riid, size, 4 , "friib" ); 
    return buffer ; 
}




/**
GMesh::getAppropriateRepeatedIdentityBuffer
---------------------------------------------

Depending on the number of ITransforms returns either

friib (numITransforms > 0)
    FaceRepeatedInstancedIdentityBuffer 
frib (numITransforms == 0)
    FaceRepeatedIdentityBuffer

Migrated here from OGeo::makeTriangulatedGeometry.

TODO: suspect might always be returning friib because even 
      for globals there may be one identity matrix ITransform.


**/

GBuffer*  GMesh::getAppropriateRepeatedIdentityBuffer()
{
    GMesh* mm = this ; 

    unsigned numITransforms = mm->getNumITransforms();
    unsigned numFaces = mm->getNumFaces();

    GBuffer* id = NULL ;  
    if(numITransforms > 0)  //  formerly 0   : HUH: perhaps should be 1,  always using friid even for globals ?
    {
        id = mm->getFaceRepeatedInstancedIdentityBuffer(); 
        assert(id);
        LOG(verbose) << "using FaceRepeatedInstancedIdentityBuffer" << " friid items " << id->getNumItems() << " numITransforms*numFaces " << numITransforms*numFaces ;     
        assert( id->getNumItems() == numITransforms*numFaces );
    }
    else
    {
        id = mm->getFaceRepeatedIdentityBuffer();
        assert(id);
        LOG(verbose) << "using FaceRepeatedIdentityBuffer" << " frid items " << id->getNumItems() << " numFaces " << numFaces ;
        assert( id->getNumItems() == numFaces );
    }
    return id ; 
}





GBuffer*  GMesh::getFaceRepeatedInstancedIdentityBuffer()
{
    if(m_facerepeated_iidentity_buffer == NULL)
    {
         m_facerepeated_iidentity_buffer = makeFaceRepeatedInstancedIdentityBuffer() ;  
    }
    return m_facerepeated_iidentity_buffer ;
}



/*

Instance Identity buffer has nodeIndex/meshIndex/boundaryIndex/sensorIndex
for all 5 volumes of each instance::

    In [3]: ii = np.load("iidentity.npy")

    In [40]: ii.reshape(-1,5,4)
    Out[40]: 
    array([[[ 3199,    47,    19,     0],
            [ 3200,    46,    20,     0],
            [ 3201,    43,    21,     3],
            [ 3202,    44,     1,     0],
            [ 3203,    45,     1,     0]],

           [[ 3205,    47,    19,     0],
            [ 3206,    46,    20,     0],
            [ 3207,    43,    21,     8],
            [ 3208,    44,     1,     0],
            [ 3209,    45,     1,     0]],

           [[ 3211,    47,    19,     0],
            [ 3212,    46,    20,     0],
            [ 3213,    43,    21,    13],
            [ 3214,    44,     1,     0],
            [ 3215,    45,     1,     0]],

    ...

    In [41]: ii.reshape(-1,5,4).shape
    Out[41]: (672, 5, 4)

    In [9]: ii.reshape(672,-1,4).shape
    Out[9]: (672, 5, 4)


    In [76]: fr = np.load("/tmp/friid.npy")

    In [80]: fr.reshape(-1,4)
    Out[80]: 
    array([[ 3199,    47,    19,     0],
           [ 3199,    47,    19,     0],
           [ 3199,    47,    19,     0],
           ..., 
           [11412,    45,     1,     0],
           [11412,    45,     1,     0],
           [11412,    45,     1,     0]], dtype=uint32)

    In [81]: fr.reshape(-1,4).shape
    Out[81]: (1967616, 4)

    In [82]: fr.reshape(672,-1,4).shape
    Out[82]: (672, 2928, 4)

    In [83]: fr[4320:5280]   # 3rd volume of 2nd instance : using face repeated IIdentity 
    Out[83]: 
    array([[3207,   43,   21,    8],
           [3207,   43,   21,    8],
           [3207,   43,   21,    8],
           ..., 
           [3207,   43,   21,    8],
           [3207,   43,   21,    8],
           [3207,   43,   21,    8]], dtype=uint32)


    In [11]: ii.reshape(672,-1,4)[1,2]    # again 3rd volume of 2nd instance : using volume level IIdentity 
    Out[11]: array([3207,   43,   21,    8], dtype=uint32)


    In [10]: ii.reshape(672,-1,4)[1]
    Out[10]: 
    array([[3205,   47,   19,    0],
           [3206,   46,   20,    0],
           [3207,   43,   21,    8],
           [3208,   44,    1,    0],
           [3209,   45,    1,    0]], dtype=uint32)





    [2015-10-09 18:39:50.180695] [0x000007fff7448031] [info]    GMesh::makeFaceRepeatedIIdentityBuffer numVolumes 5 numFaces (sum of faces in numVolumes)2928 numITransforms 672 numRepeatedIdentity 1967616
     s 0 nf 720  i0 0:720  i1 2928:3648   il 1964688:1965408 
     s 1 nf 672  i0 720:1392  i1 3648:4320   il 1965408:1966080 
     s 2 nf 960  i0 1392:2352  i1 4320:5280   il 1966080:1967040 
     s 3 nf 480  i0 2352:2832  i1 5280:5760   il 1967040:1967520 
     s 4 nf  96  i0 2832:2928  i1 5760:5856   il 1967520:1967616 
     ----- 2928 


*/


GBuffer* GMesh::makeFaceRepeatedIdentityBuffer()
{
    unsigned int numITransforms = getNumITransforms() ;
    if(numITransforms > 1)  // formerly 0, but have moved to assigning identity buffer to non-instanced
    {
        LOG(warning) << "GMesh::makeFaceRepeatedIdentityBuffer only relevant to non-instanced meshes " ;
        return NULL ; 
    }
    unsigned int numVolumes = getNumVolumes();
    unsigned int numFaces = getNumFaces() ;

    LOG(info) << "GMesh::makeFaceRepeatedIdentityBuffer"
              << " numVolumes " << numVolumes 
              << " numFaces (sum of faces in numVolumes)" << numFaces 
               ; 

    assert(m_nodeinfo_buffer->getNumItems() == numVolumes);

    guint4* nodeinfo = getNodeInfo();


    // check nodeinfo sum of per-volume faces matches expectation
    unsigned int nftot(0);
    for(unsigned int s=0 ; s < numVolumes ; s++)
    {
        unsigned int nf = (nodeinfo + s)->x ;
        nftot += nf ;
    }
    printf(" ----- %d \n", nftot);
    assert( numFaces == nftot );


    // duplicate nodeinfo for each volume out to each face
    unsigned int offset(0);
    guint4* rid = new guint4[numFaces] ;
    for(unsigned int s=0 ; s < numVolumes ; s++)
    {   
        guint4 sid = m_identity[s]  ;  
        unsigned int nf = (nodeinfo + s)->x ;
        for(unsigned int f=0 ; f < nf ; ++f) rid[offset+f] = sid ; 
        offset += nf ; 
    }  
    
    unsigned int size = sizeof(guint4) ;
    GBuffer* buffer = new GBuffer( size*numFaces, (void*)rid, size, 4 , "frib" ); 
    return buffer ; 
}




GBuffer*  GMesh::getFaceRepeatedIdentityBuffer()
{
    if(m_facerepeated_identity_buffer == NULL)
    {
         m_facerepeated_identity_buffer = makeFaceRepeatedIdentityBuffer() ;  
    }
    return m_facerepeated_identity_buffer ;
}



/*

GBuffer* GMesh::loadAnalyticGeometryBuffer(const char* path)
{
    GBuffer* partBuf = GBuffer::load<float>(path);
    return partBuf ; 
}

GBuffer*  GMesh::getAnalyticGeometryBuffer()
{
    assert(0);
    //if(m_analytic_geometry_buffer == NULL)
    //{
    //    m_analytic_geometry_buffer = loadAnalyticGeometryBuffer() ; // FIX: relying on default temporary path
    //}
    //return m_analytic_geometry_buffer ;
    return NULL ; 
}

*/



void GMesh::explodeZVertices(float zoffset, float zcut)
{
    unsigned int noffset(0);
    for(unsigned int i=0 ; i < m_num_vertices ; i++)
    {
        gfloat3* v = m_vertices + i  ;
        if( v->z > zcut )
        {
            noffset += 1 ; 
            v->z += zoffset ; 
            printf("GMesh::explodeZVertices %6d : %10.4f %10.4f %10.4f \n", i, v->x, v->y, v->z );
        }
    }
    LOG(info) << "GMesh::explodeZVertices" 
              << " noffset " << noffset 
              << " zoffset " << zoffset 
              << " zcut " << zcut 
              ;
}


unsigned int GMesh::findContainer(gfloat3 p)
{
   // find volumes that contains the point
   // returning the index of the volume with the smallest extent 
   // or 0 if none found
   //
    unsigned int container(0);
    float cext(FLT_MAX) ; 

    for(unsigned int index=0 ; index < m_num_volumes ; index++)
    {
         gfloat4 ce = m_center_extent[index] ;
         gfloat3 hi(ce.x + ce.w, ce.y + ce.w, ce.z + ce.w );
         gfloat3 lo(ce.x - ce.w, ce.y - ce.w, ce.z - ce.w );

         if( 
              p.x > lo.x && p.x < hi.x  &&
              p.y > lo.y && p.y < hi.y  &&
              p.z > lo.z && p.z < hi.z 
           )
          {
               //printf("GMesh::findContainer %d   %10.4f %10.4f %10.4f %10.4f  \n", index, ce.x, ce.y, ce.z, ce.w  );
               if(ce.w < cext)
               {
                   cext = ce.w ; 
                   container = index ; 
               }
          }
    }
    return container ; 
}



/**
GMesh::isGlobalInstance
------------------------------

GlobalInstance meshes contain the global geometry treated as an ordinary instanced mesh
unlike the special treatment of the index 0 global GMesh.

**/
bool GMesh::isGlobalInstance() const 
{
    return m_globalinstance ; 
}
void GMesh::setGlobalInstance(bool globalinstance)
{
    m_globalinstance = globalinstance ; 
}








template <typename T>
void GMesh::setMeta(const char* key, T value)
{
    assert( m_itransforms_buffer ); 
    m_itransforms_buffer->setMeta<T>(key, value);
}
template <typename T>
T GMesh::getMeta(const char* key, const char* fallback) const 
{
    assert( m_itransforms_buffer ); 
    return m_itransforms_buffer->getMeta<T>(key, fallback);
}



template GGEO_API void GMesh::setMeta(const char* key, int value);
template GGEO_API void GMesh::setMeta(const char* key, float value);
template GGEO_API void GMesh::setMeta(const char* key, std::string value);

template GGEO_API int GMesh::getMeta(const char* key, const char* fallback) const ;
template GGEO_API float GMesh::getMeta(const char* key, const char* fallback) const ;
template GGEO_API std::string GMesh::getMeta(const char* key, const char* fallback) const ;





 
