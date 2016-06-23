
#include <vector>
#include <climits>
#include <iostream>
#include <iomanip>

#include <boost/filesystem.hpp>

// npy-
#include "Timer.hpp"
#include "NSensor.hpp"

// opticks-
#include "Opticks.hh"
#include "OpticksConst.hh"
#include "OpticksResource.hh"


#include "GGeo.hh"
#include "GSolid.hh"
#include "GParts.hh"
#include "GMergedMesh.hh"


#include "PLOG.hh"


namespace fs = boost::filesystem;


bool GMergedMesh::isSkip()
{
   return m_geocode == OpticksConst::GEOCODE_SKIP ;  
}
bool GMergedMesh::isAnalytic()
{
   return m_geocode == OpticksConst::GEOCODE_ANALYTIC ;  
}
bool GMergedMesh::isTriangulated()
{
   return m_geocode == OpticksConst::GEOCODE_TRIANGULATED ;  
}



GMergedMesh::GMergedMesh(unsigned int index)
       : 
       GMesh(index, NULL, 0, NULL, 0, NULL, NULL),
       m_cur_vertices(0),
       m_cur_faces(0),
       m_cur_solid(0),
       m_cur_base(NULL)
{
} 

void GMergedMesh::setCurrentBase(GNode* base)
{
    m_cur_base = base ; 
}
GNode* GMergedMesh::getCurrentBase()
{
    return m_cur_base ; 
}


bool GMergedMesh::isGlobal()
{
    return m_cur_base == NULL ; 
}
bool GMergedMesh::isInstanced()
{
    return m_cur_base != NULL ; 
}











GMergedMesh* GMergedMesh::combine(unsigned int index, GMergedMesh* mm, GSolid* solid)
{
    std::vector<GSolid*> solids ; 
    solids.push_back(solid);
    return combine(index, mm, solids );
}

GMergedMesh* GMergedMesh::combine(unsigned int index, GMergedMesh* mm, std::vector<GSolid*>& solids)
{
    unsigned int verbosity = mm ? mm->getVerbosity() : 0 ;     
    LOG(info) << "GMergedMesh::combine"
              << " making new mesh "
              << " index " << index 
              << " solids " << solids.size()
              << " verbosity " << verbosity 
               ; 

    std::vector<GParts*> analytic ; 
    GMergedMesh* com = new GMergedMesh( index ); 
    com->setVerbosity(verbosity);

    if(mm)
    {
        com->countMergedMesh(mm, true);
        GParts* pts = mm->getParts();

        if(pts)
            analytic.push_back(pts);
        else
            LOG(warning) << "GMergedMesh::combine mm has no analytic GParts attached " ; 

    }

    typedef std::vector<GSolid*> VS ; 
    for(VS::const_iterator it=solids.begin() ; it != solids.end() ; it++)
    {
        GSolid* solid = *it ; 
        com->countSolid(solid, true);

        GParts* pts = solid->getParts();
        if(pts)
            analytic.push_back(pts);
        else
            LOG(warning) << "GMergedMesh::combine solid has no analytic GParts attached " ; 

    } 

    com->allocate(); 
 
    if(mm)
    {
        com->mergeMergedMesh(mm, true);
    } 

    for(VS::const_iterator it=solids.begin() ; it != solids.end() ; it++)
    {
        GSolid* solid = *it ; 
        com->mergeSolid(solid, true);
    } 

    com->updateBounds();



    unsigned int ncomp = solids.size() + ( mm ? 1 : 0 ) ;

    if(analytic.size() == ncomp)
    {
        GParts*      anl = GParts::combine(analytic);
        com->setParts(anl);
    }
    else
    {
        LOG(warning) << "GMergedMesh::combine CANNOT combine analytic parts are incomplete " 
                     << " ncomp " << ncomp 
                     << " nanalytic " << analytic.size()
                     ;
    }

    return com ; 
}



GMergedMesh* GMergedMesh::create(unsigned int index, GGeo* ggeo, GNode* base)
{
    // needs GGeo just to access root node

    Timer t("GMergedMesh::create") ; 
    t.setVerbose(false);
    t.start();

    GMergedMesh* mm = new GMergedMesh( index ); 
    mm->setVerbosity(ggeo->getMeshVerbosity());

    if(base == NULL)  // non-instanced global transforms
    {
        mm->setCurrentBase(NULL);
        base = static_cast<GNode*>(ggeo->getSolid(0)); 
        unsigned int numMeshes = ggeo->getNumMeshes();
        assert(numMeshes < 500 );
    }
    else   // instances transforms, with transform heirarchy split at the base 
    {
        mm->setCurrentBase(base);
    }

    LOG(info)<<"GMergedMesh::create"
             << " index " << index 
             << " from base " << base->getName() ;
             ; 

    // 1st pass traversal : counts vertices and faces

    mm->traverse( base, 0, PASS_COUNT );  

    t("1st pass traverse");

    // allocate space for flattened arrays

    LOG(info) << "GMergedMesh::create" 
              << " index " << index 
              << " numVertices " << mm->getNumVertices()
              << " numFaces " << mm->getNumFaces()
              << " numSolids " << mm->getNumSolids()
              << " numSolidsSelected " << mm->getNumSolidsSelected()
              ;

    mm->allocate(); 

    // 2nd pass traversal : merge copy GMesh into GMergedMesh 

    mm->traverse( base, 0, PASS_MERGE );  
    t("2nd pass traverse");

    mm->updateBounds();

    t("updateBounds");

    t.stop();
    //t.dump();

    return mm ;
}



/*
Huh ? why do OAV and IAV give a number of solids of 1 whereas all others give zero ?  maybe related to mesh fixing 

[2015-Nov-04 17:28:12.203204]:info: GMergedMesh::count  selected false mesh.nsolid 0 mesh.name near_pool_iws_box0xc288ce8 num_solids 0 num_solids_selected 0
[2015-Nov-04 17:28:12.203320]:info: GMergedMesh::count  selected true mesh.nsolid 0 mesh.name ade0xc2a7438 num_solids 0 num_solids_selected 0
[2015-Nov-04 17:28:12.203430]:info: GMergedMesh::count  selected true mesh.nsolid 0 mesh.name sst0xbf4b060 num_solids 0 num_solids_selected 0
[2015-Nov-04 17:28:12.203538]:info: GMergedMesh::count  selected true mesh.nsolid 0 mesh.name oil0xbf5ed48 num_solids 0 num_solids_selected 0
[2015-Nov-04 17:28:12.203648]:info: GMergedMesh::count  selected true mesh.nsolid 1 mesh.name oav0xc2ed7c8 num_solids 1 num_solids_selected 0
[2015-Nov-04 17:28:12.203758]:info: GMergedMesh::count  selected true mesh.nsolid 0 mesh.name lso0xc028a38 num_solids 1 num_solids_selected 1
[2015-Nov-04 17:28:12.203866]:info: GMergedMesh::count  selected true mesh.nsolid 1 mesh.name iav0xc346f90 num_solids 2 num_solids_selected 1
[2015-Nov-04 17:28:12.203974]:info: GMergedMesh::count  selected true mesh.nsolid 0 mesh.name gds0xc28d3f0 num_solids 2 num_solids_selected 2
[2015-Nov-04 17:28:12.204082]:info: GMergedMesh::count  selected true mesh.nsolid 0 mesh.name OcrGdsInIav0xc405b10 num_solids 2 num_solids_selected 2
[2015-Nov-04 17:28:12.204195]:info: GMergedMesh::count  selected true mesh.nsolid 0 mesh.name IavTopHub0xc405968 num_solids 2 num_solids_selected 2
[2015-Nov-04 17:28:12.204310]:info: GMergedMesh::count  selected true mesh.nsolid 0 mesh.name CtrGdsOflBotClp0xbf5dec0 num_solids 2 num_solids_selected 2
...
[2015-Nov-04 17:28:12.397076]:info: GMergedMesh::count  selected true mesh.nsolid 0 mesh.name MOFTTopCover0xc047878 num_solids 2 num_solids_selected 2
[2015-Nov-04 17:28:12.397190]:info: GMergedMesh::count  selected true mesh.nsolid 0 mesh.name ade0xc2a7438 num_solids 2 num_solids_selected 2
[2015-Nov-04 17:28:12.397297]:info: GMergedMesh::count  selected true mesh.nsolid 0 mesh.name sst0xbf4b060 num_solids 2 num_solids_selected 2
[2015-Nov-04 17:28:12.397406]:info: GMergedMesh::count  selected true mesh.nsolid 0 mesh.name oil0xbf5ed48 num_solids 2 num_solids_selected 2
[2015-Nov-04 17:28:12.397512]:info: GMergedMesh::count  selected true mesh.nsolid 1 mesh.name oav0xc2ed7c8 num_solids 3 num_solids_selected 2
[2015-Nov-04 17:28:12.397619]:info: GMergedMesh::count  selected true mesh.nsolid 0 mesh.name lso0xc028a38 num_solids 3 num_solids_selected 3
[2015-Nov-04 17:28:12.397725]:info: GMergedMesh::count  selected true mesh.nsolid 1 mesh.name iav0xc346f90 num_solids 4 num_solids_selected 3
[2015-Nov-04 17:28:12.397831]:info: GMergedMesh::count  selected true mesh.nsolid 0 mesh.name gds0xc28d3f0 num_solids 4 num_solids_selected 4
[2015-Nov-04 17:28:12.397936]:info: GMergedMesh::count  selected true mesh.nsolid 0 mesh.name OcrGdsInIav0xc405b10 num_solids 4 num_solids_selected 4
[2015-Nov-04 17:28:12.398048]:info: GMergedMesh::count  selected true mesh.nsolid 0 mesh.name IavTopHub0xc405968 num_solids 4 num_solids_selected 4

*/



// NB what is appropriate for a merged mesh is not for a mesh ... wrt counting solids
// so cannot lump the below together using GMesh base class

void GMergedMesh::countMergedMesh( GMergedMesh*  other, bool selected)
{
    unsigned int nsolid = other->getNumSolids();

    m_num_solids += nsolid ;

    if(selected)
    {
        m_num_solids_selected += 1 ;
        countMesh( other ); 
    }

    if(m_verbosity > 1)
    LOG(info) << "GMergedMesh::count other GMergedMesh  " 
              << " selected " << selected
              << " num_solids " << m_num_solids 
              << " num_solids_selected " << m_num_solids_selected 
              ;
}

void GMergedMesh::countSolid( GSolid* solid, bool selected)
{
    GMesh* mesh = solid->getMesh();

    m_num_solids += 1 ; 

    if(selected)
    {
        m_num_solids_selected += 1 ;
        countMesh( mesh ); 
    }

    if(m_verbosity > 1)
    LOG(info) << "GMergedMesh::count GSolid " 
              << " selected " << selected
              << " num_solids " << m_num_solids 
              << " num_solids_selected " << m_num_solids_selected 
              ;
}

void GMergedMesh::countMesh( GMesh* mesh )
{
    unsigned int nface = mesh->getNumFaces();
    unsigned int nvert = mesh->getNumVertices();
    unsigned int meshIndex = mesh->getIndex();

    m_num_vertices += nvert ;
    m_num_faces    += nface ; 
    m_mesh_usage[meshIndex] += 1 ;  // which meshes contribute to the mergedmesh
}


void GMergedMesh::mergeMergedMesh( GMergedMesh* other, bool selected )
{
    // solids are present irrespective of selection as prefer absolute solid indexing 

    unsigned int nsolid = other->getNumSolids();

    if(m_verbosity > 1)
    LOG(info) << "GMergedMesh::mergeMergedMesh"
              << " m_cur_solid " << m_cur_solid
              << " m_cur_vertices " << m_cur_vertices
              << " m_cur_faces " << m_cur_faces
              << " other nsolid " << nsolid 
              << " selected " << selected
              ; 


    for(unsigned int i=0 ; i < nsolid ; i++)
    {
        gbbox bb = other->getBBox(i) ;
        gfloat4 ce = other->getCenterExtent(i) ;

        if(m_verbosity > 2)
        LOG(info) << "GMergedMesh::mergeMergedMesh"
                   << " m_cur_solid " << m_cur_solid  
                   << " i " << i
                   << " ce " <<  ce.description() 
                   << " bb " <<  bb.description() 
                   ;

        m_bbox[m_cur_solid] = bb ;  
        m_center_extent[m_cur_solid] = ce ;
        m_nodeinfo[m_cur_solid] = other->getNodeInfo(i) ; 
        m_identity[m_cur_solid] = other->getIdentity(i) ; 
        m_meshes[m_cur_solid] = other->getMeshIndice(i) ; 

        memcpy( getTransform(m_cur_solid), other->getTransform(i), 16*sizeof(float) ); 

        m_cur_solid += 1 ; 
    }

    unsigned int nvert = other->getNumVertices();
    unsigned int nface = other->getNumFaces();

    gfloat3* vertices = other->getVertices();
    gfloat3* normals = other->getNormals();
    guint3*  faces = other->getFaces();

    unsigned int* node_indices = other->getNodes();
    unsigned int* boundary_indices = other->getBoundaries();
    unsigned int* sensor_indices = other->getSensors();

    assert(node_indices);
    assert(boundary_indices);
    assert(sensor_indices);

    if(selected)
    {
        for(unsigned int i=0 ; i<nvert ; ++i )
        {
            m_vertices[m_cur_vertices+i] = vertices[i] ; 
            m_normals[m_cur_vertices+i] = normals[i] ; 
        }

        for(unsigned int i=0 ; i<nface ; ++i )
        {
            m_faces[m_cur_faces+i].x = faces[i].x + m_cur_vertices ;  
            m_faces[m_cur_faces+i].y = faces[i].y + m_cur_vertices ;  
            m_faces[m_cur_faces+i].z = faces[i].z + m_cur_vertices ;  

            m_nodes[m_cur_faces+i]      = node_indices[i] ;
            m_boundaries[m_cur_faces+i] = boundary_indices[i] ;
            m_sensors[m_cur_faces+i]    = sensor_indices[i] ;
        }

        // offset within the flat arrays
        m_cur_vertices += nvert ;
        m_cur_faces    += nface ;
    }

}

void GMergedMesh::mergeSolid( GSolid* solid, bool selected )
{
    GMesh* mesh = solid->getMesh();
    unsigned int nvert = mesh->getNumVertices();
    unsigned int nface = mesh->getNumFaces();
    guint4 _identity = solid->getIdentity();

    GNode* base = getCurrentBase();
    GMatrixF* transform = base ? solid->getRelativeTransform(base) : solid->getTransform() ;    
    gfloat3* vertices = mesh->getTransformedVertices(*transform) ;

    // needs to be outside the selection branch for the all solid center extent
    gbbox bb = GMesh::findBBox(vertices, nvert) ;

   if(m_verbosity > 1)
   {

        const char* pvn = solid->getPVName() ;
        const char* lvn = solid->getLVName() ;

        LOG(info) << "GMergedMesh::mergeSolid" 
                  << " m_cur_solid " << m_cur_solid
                  << " idx " << solid->getIndex()
                  << " id " << _identity.description()
                  << " pv " << ( pvn ? pvn : "-" )
                  << " lv " << ( lvn ? lvn : "-" )
                  << " bb " << bb.description()
                  ;
        transform->Summary("GMergedMesh::mergeSolid transform");
   }   


    unsigned int boundary = solid->getBoundary();
    NSensor* sensor = solid->getSensor();

    unsigned int nodeIndex = solid->getIndex();
    unsigned int meshIndex = mesh->getIndex();
    unsigned int sensorIndex = NSensor::RefIndex(sensor) ; 
    assert(_identity.x == nodeIndex);
    assert(_identity.y == meshIndex);
    assert(_identity.z == boundary);
    //assert(_identity.w == sensorIndex);   this is no longer the case, now require SensorSurface in the identity
    
    LOG(debug) << "GMergedMesh::mergeSolid"
              << " m_cur_solid " << m_cur_solid 
              << " nodeIndex " << nodeIndex
              << " boundaryIndex " << boundary
              << " sensorIndex " << sensorIndex
              << " sensor " << ( sensor ? sensor->description() : "NULL" )
              ;

    GNode* parent = solid->getParent();
    unsigned int parentIndex = parent ? parent->getIndex() : UINT_MAX ;


    m_bbox[m_cur_solid] = bb ;  
    m_center_extent[m_cur_solid] = bb.center_extent() ;

    float* dest = getTransform(m_cur_solid);
    assert(dest);

    transform->copyTo(dest);
    m_meshes[m_cur_solid] = meshIndex ; 

    // face and vertex counts must use same selection as above to be usable 
    // with the above filled vertices and indices 

    m_nodeinfo[m_cur_solid].x = selected ? nface : 0 ; 
    m_nodeinfo[m_cur_solid].y = selected ? nvert : 0 ; 
    m_nodeinfo[m_cur_solid].z = nodeIndex ;  
    m_nodeinfo[m_cur_solid].w = parentIndex ; 

    if(isGlobal())
    {
         if(nodeIndex != m_cur_solid)
             LOG(fatal) << "GMergedMesh::mergeSolid mismatch " 
                        <<  " nodeIndex " << nodeIndex 
                        <<  " m_cur_solid " << m_cur_solid
                        ; 

         //assert(nodeIndex == m_cur_solid);  // trips ggv-pmt still needed ?
    } 

    m_identity[m_cur_solid] = _identity ; 

    m_cur_solid += 1 ;    // irrespective of selection, as prefer absolute solid indexing 



    if(selected)
    {
        gfloat3* normals = mesh->getTransformedNormals(*transform);  
        guint3* faces = mesh->getFaces();

        for(unsigned int i=0 ; i<nvert ; ++i )
        {
            m_vertices[m_cur_vertices+i] = vertices[i] ; 
            m_normals[m_cur_vertices+i] = normals[i] ; 
        }

        // TODO: consolidate into uint4 (with one spare)
        unsigned int* node_indices = solid->getNodeIndices();
        unsigned int* boundary_indices = solid->getBoundaryIndices();
        unsigned int* sensor_indices = solid->getSensorIndices();
        assert(node_indices);
        assert(boundary_indices);
        assert(sensor_indices);

        // offset the vertex indices as are combining all meshes into single vertex list 
        for(unsigned int i=0 ; i<nface ; ++i )
        {
            m_faces[m_cur_faces+i].x = faces[i].x + m_cur_vertices ;  
            m_faces[m_cur_faces+i].y = faces[i].y + m_cur_vertices ;  
            m_faces[m_cur_faces+i].z = faces[i].z + m_cur_vertices ;  

            m_nodes[m_cur_faces+i]      = node_indices[i] ;
            m_boundaries[m_cur_faces+i] = boundary_indices[i] ;
            m_sensors[m_cur_faces+i]    = sensor_indices[i] ;
        }

        // offset within the flat arrays
        m_cur_vertices += nvert ;
        m_cur_faces    += nface ;
    }
}


void GMergedMesh::traverse( GNode* node, unsigned int depth, unsigned int pass)
{
    GSolid* solid = dynamic_cast<GSolid*>(node) ;

    // using repeat index labelling in the tree
    //  bool repsel = getIndex() == -1 || solid->getRepeatIndex() == getIndex() ;

    int idx = getIndex() ;
    unsigned int uidx = idx > -1 ? idx : UINT_MAX ; 
    unsigned int ridx = solid->getRepeatIndex() ;
    bool repsel =  idx == -1 || ridx == uidx ;
    bool selected = solid->isSelected() && repsel ;

    switch(pass)
    {
       case PASS_COUNT:    countSolid(solid, selected)  ;break;
       case PASS_MERGE:    mergeSolid(solid, selected)  ;break;
               default:    assert(0)                    ;break;
    }

    for(unsigned int i = 0; i < node->getNumChildren(); i++) traverse(node->getChild(i), depth + 1, pass);
}



void GMergedMesh::reportMeshUsage(GGeo* ggeo, const char* msg)
{
     LOG(info) << msg ; 
     typedef std::map<unsigned int, unsigned int>::const_iterator MUUI ; 

     unsigned int tv(0) ; 
     for(MUUI it=m_mesh_usage.begin() ; it != m_mesh_usage.end() ; it++)
     {
         unsigned int meshIndex = it->first ; 
         unsigned int nodeCount = it->second ; 
 
         GMesh* mesh = ggeo->getMesh(meshIndex);
         const char* meshName = mesh->getName() ; 
         unsigned int nv = mesh->getNumVertices() ; 
         unsigned int nf = mesh->getNumFaces() ; 

         printf("  %4d (v%5d f%5d) : %6d : %7d : %s \n", meshIndex, nv, nf, nodeCount, nodeCount*nv, meshName);

         tv += nodeCount*nv ; 
     }
     printf(" tv : %7d \n", tv);
}




GMergedMesh* GMergedMesh::load(Opticks* opticks, unsigned int ridx, const char* version)
{
    std::string mmpath = opticks->getResource()->getMergedMeshPath(ridx);
    GMergedMesh* mm = GMergedMesh::load(mmpath.c_str(), ridx, version);
    return mm ; 
}

GMergedMesh* GMergedMesh::load(const char* dir, unsigned int index, const char* version)
{
    GMergedMesh* mm(NULL);
    fs::path cachedir(dir);
    if(!fs::exists(cachedir))
    {
        LOG(warning) << "GMergedMesh::load directory DOES NOT EXIST " <<  dir ;
    }
    else
    {
        mm = new GMergedMesh(index);
        if(index == 0) mm->setVersion(version);  // mesh versioning applies to  global buffer 
        mm->loadBuffers(dir);
    }
    return mm ; 
}


void GMergedMesh::dumpSolids(const char* msg)
{
    LOG(info) << msg ; 
    for(unsigned int index=0 ; index < getNumSolids() ; ++index)
    {
        //if(index % 1000 != 0) continue ; 
        gfloat4 ce = getCenterExtent(index) ;
        gbbox bb = getBBox(index) ; 

        std::cout 
             << std::setw(5)  << index         
             << " ce " << std::setw(64) << ce.description()       
             << " bb " << std::setw(64) << bb.description()       
             << std::endl 
             ;

        //printf("  %u :    center %10.3f %10.3f %10.3f   extent %10.3f \n", index, ce.x, ce.y, ce.z, ce.w ); 
    }
}

float* GMergedMesh::getModelToWorldPtr(unsigned int index)
{
    return index == 0 ? GMesh::getModelToWorldPtr(0) : NULL ;
}


