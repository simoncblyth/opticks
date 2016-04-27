#include "AssimpNode.hh"
#include "AssimpTree.hh"
#include "AssimpCommon.hh"
#include "assert.h"
#include "stdio.h"
#include <assimp/scene.h>

AssimpNode::~AssimpNode()
{
}

AssimpNode::AssimpNode(std::vector<aiNode*> nodepath, AssimpTree* tree) 
   : 
   m_index(0),
   m_depth(0),
   m_raw(nodepath.back()),
   m_nodepath(nodepath),
   m_digest(0),
   m_pdigest(0),

   m_meshes(NULL),
   m_numMeshes(0),

   m_low(NULL),
   m_high(NULL),
   m_center(NULL),
   m_extent(NULL),

   m_tree(tree),
   m_parent(NULL)
{

   unsigned int leafdepth = nodepath.size() ;
   m_digest = hash(0, leafdepth);
   if( leafdepth > 2 ) m_pdigest = hash(0,leafdepth-2);

   setDepth(leafdepth);
}



aiMatrix4x4 AssimpNode::getGlobalTransform()
{
    // applies all transforms from the raw tree
    aiMatrix4x4 transform ;
    for(unsigned int i=0 ; i < m_nodepath.size() ; ++i )
    {  
        aiNode* node = m_nodepath[i] ;
        transform *= node->mTransformation ; 
    }
    return transform ; 
}

aiMatrix4x4 AssimpNode::getTransform()
{
   return m_transform ;   // globalTransform set by copyMeshes
}

aiMatrix4x4 AssimpNode::getLevelTransform(int level)
{
   if(level < 0) level += m_nodepath.size();
   assert(level < int(m_nodepath.size()));
   aiNode* node = m_nodepath[level] ;
   return node->mTransformation ;
}



aiNode* AssimpNode::getRawNode()
{
   return m_raw ; 
}

aiNode* AssimpNode::getRawNode(unsigned int iback)
{
    unsigned int size = m_nodepath.size() ;
    unsigned int index = size - 1 - iback ; 
    return index < size ?  m_nodepath[index] : NULL  ; 
}

const char* AssimpNode::getName() 
{
    return m_raw->mName.C_Str();
}
const char* AssimpNode::getName(unsigned int iback) 
{
    aiNode* raw = getRawNode(iback);
    return raw ? raw->mName.C_Str() : "-" ;
}


std::size_t AssimpNode::hash(unsigned int pyfirst, unsigned int pylast)
{
    unsigned int size = m_nodepath.size() ;
    //printf("AssimpNode::hash pyfirst %u pylast %u size %u \n", pyfirst, pylast, size); 
    assert(pyfirst <= size); 
    assert(pylast  <= size);

    std::size_t h = 0;
    for(unsigned int i=pyfirst ; i < pylast ; ++i )
    {  
        aiNode* node = m_nodepath[i] ;
        std::size_t inode = (std::size_t)node ; 
        // come up with a more standard hashing 
        h ^= inode + 0x9e3779b9 + ( h  << 6) + (h >> 2);
    }
    return h ;
}


void AssimpNode::traverse()
{
   summary("AssimpNode::traverse");
   for(unsigned int i=0 ; i < getNumChildren() ; i++ ) getChild(i)->traverse(); 
}

void AssimpNode::summary(const char* msg)
{
    unsigned int nchild = getNumChildren();
    unsigned int nmesh = getNumMeshes() ;
    printf("%s index %5d depth %2d nchild %4d nmesh %d digest %20zu pdigest %20zu name %s  \n", msg, m_index, m_depth, nchild, nmesh, m_digest, m_pdigest, getName() );
    bounds();
}


void AssimpNode::dump()
{
    unsigned int nchild = getNumChildren();
    //unsigned int nprog = progeny();
    unsigned int nmesh = getNumMeshes() ;

    if(nchild > 8)
    {
        for(unsigned int i = 0; i < nmesh ; i++)
        {   
            aiMesh* rawmesh = getRawMesh(i);

            printf("AssimpNode::dump  i %d \n", i);

            dumpMesh(rawmesh);

            aiMesh* mesh = getMesh(i);

            dumpMesh(mesh);
        }   
    }
}


void AssimpNode::bounds(const char* msg)
{
    if(m_center) printf("%s cen  %10.3f %10.3f %10.3f  \n", msg, m_center->x, m_center->y, m_center->z );
    if(m_low)    printf("%s low  %10.3f %10.3f %10.3f  \n", msg, m_low->x, m_low->y, m_low->z );
    if(m_high)   printf("%s high %10.3f %10.3f %10.3f \n",  msg, m_high->x, m_high->y, m_high->z );
    if(m_extent) printf("%s ext %10.3f %10.3f %10.3f \n", msg, m_extent->x, m_extent->y, m_extent->z);
}



void AssimpNode::copyMeshes(aiMatrix4x4 transform)
{

     m_transform = transform ; 

     m_numMeshes = m_raw->mNumMeshes ; 

     m_meshes = new aiMesh*[m_numMeshes];   // array of pointers to meshes

     aiVector3D  low( 1e10f, 1e10f, 1e10f);
     aiVector3D high( -1e10f, -1e10f, -1e10f);

     // typically only one mesh per node for COLLADA, but retaining generality
     for(unsigned int i = 0; i < m_numMeshes ; i++)  
     {
          aiMesh* src = getRawMesh(i);
          m_meshes[i] = new aiMesh ; 
          copyMesh(m_meshes[i], src, m_transform);  // from   dst <-- transform * src 
          meshBounds(m_meshes[i], low, high );
     }

     if(m_numMeshes > 0)
     {
         m_low  = new aiVector3D(low);
         m_high = new aiVector3D(high);
         m_center = new aiVector3D((high+low)/2.f);
         m_extent = new aiVector3D(high-low);
     }
}


void AssimpNode::updateBounds()
{
    aiVector3D  low( 1e10f, 1e10f, 1e10f);
    aiVector3D high( -1e10f, -1e10f, -1e10f);

    // containment and lv/pv alternation 
    // should mean self and children is enough to always get some bounds ? 
    //
    updateBounds(low,high);

    for(unsigned int i=0 ; i < getNumChildren() ; i++ ) getChild(i)->updateBounds(low, high); 

    delete m_low ; 
    m_low  = new aiVector3D(low);

    delete m_high ;
    m_high = new aiVector3D(high);

    delete m_center ;
    m_center = new aiVector3D((low + high)/2.f);

}


void AssimpNode::updateBounds(aiVector3D& low, aiVector3D& high)
{
    if(m_low && m_high)
    {
        low.x = std::min( low.x, m_low->x);
        low.y = std::min( low.y, m_low->y);
        low.z = std::min( low.z, m_low->z);

        high.x = std::max( high.x, m_high->x);
        high.y = std::max( high.y, m_high->y);
        high.z = std::max( high.z, m_high->z);
   } 
}



unsigned int AssimpNode::getNumMeshes()
{
    return m_numMeshes ; 
}
unsigned int AssimpNode::getNumMeshesRaw()
{
    return m_raw->mNumMeshes ; 
}



aiMesh* AssimpNode::getRawMesh(unsigned int localMeshIndex)
{
     unsigned int meshIndex = m_raw->mMeshes[localMeshIndex];
     aiMesh* mesh = m_tree->getRawMesh(meshIndex);
     return mesh ;
}

unsigned int AssimpNode::getMaterialIndex(unsigned int localMeshIndex)
{
    aiMesh* mesh = getRawMesh(localMeshIndex); 
    return mesh->mMaterialIndex ;     
}

char* AssimpNode::getMaterialName(unsigned int localMeshIndex)
{
    unsigned int materialIndex = getMaterialIndex(localMeshIndex);
    return m_tree->getMaterialName(materialIndex); 
}



char* AssimpNode::getDescription(const char* label)
{
    AssimpNode* pnode = getParent();
    if(!pnode) pnode = this ; 

    char* mtn   = getMaterialName() ;
    char* mtn_p = pnode->getMaterialName() ;

    char desc[1024];
    snprintf(desc, 1024,"%s\n    pv   %5u [%4u] (%2u,%3u)%-50s %s\n    pv_p %5u [%4u] (%2u,%3u)%-50s %s\n    lv   %5u [%4u] (      )%-50s %s\n    lv_p %5u [%4u] (      )%-50s %s\n", 
          label,

          getIndex(),
          getNumChildren(),
          getMaterialIndex(), 
          getMeshIndex(), 
          mtn,
          getName(1), 

          pnode->getIndex(),
          pnode->getNumChildren(),
          pnode->getMaterialIndex(), 
          pnode->getMeshIndex(), 
          mtn_p, 
          pnode->getName(1), 

          getIndex(),
          getNumChildren(),
          "",
          getName(0),

          pnode->getIndex(),
          pnode->getNumChildren(),
          "",
          pnode->getName(0)
          );

     free(mtn);
     free(mtn_p);
     return strdup(desc);
}



unsigned int AssimpNode::getMeshIndex(unsigned int localMeshIndex)
{
    unsigned int meshIndex = m_raw->mMeshes[localMeshIndex];
    return meshIndex ; 
}

aiMesh* AssimpNode::getMesh(unsigned int index)
{
    return index < m_numMeshes ? m_meshes[index] : NULL ;
}


void AssimpNode::ancestors()
{
   AssimpNode* node = this ; 
   unsigned int count = 0 ;
   while( node )
   {
       printf("AssimpNode::ancestors count %2d index %4d  n0:%s n1:%s n2:%s \n", count, node->getIndex(), node->getName(0), node->getName(1), node->getName(2) );
       node = node->getParent();
       count++ ;
   }
}

unsigned int AssimpNode::progeny()
{
   //printf("AssimpNode::progeny %s \n", name());
   unsigned int nchild = getNumChildren();
   unsigned int tot = nchild ; 
   for(unsigned int i=0 ; i < nchild ; i++ ) tot += getChild(i)->progeny(); 
   return tot ; 
}



