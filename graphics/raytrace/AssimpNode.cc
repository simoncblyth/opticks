#include "AssimpNode.hh"
#include "AssimpTree.hh"
#include "AssimpCommon.hh"
#include "assert.h"

#include <assimp/scene.h>

AssimpNode::~AssimpNode()
{
}

AssimpNode::AssimpNode(std::vector<aiNode*> nodepath, AssimpTree* tree) 
   : 
   m_parent(NULL),
   m_tree(tree),
   m_nodepath(nodepath),
   m_raw(nodepath.back()),
   m_index(0),
   m_digest(0),
   m_pdigest(0),
   m_meshes(NULL),
   m_numMeshes(0),
   m_low(NULL),
   m_high(NULL),
   m_center(NULL),
   m_extent(NULL)
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



std::size_t AssimpNode::getDigest()
{
   return m_digest ;
}
std::size_t AssimpNode::getParentDigest()
{
   return m_pdigest ;
}
aiNode* AssimpNode::getRawNode(){
   return m_raw ; 
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


const char* AssimpNode::getName() {
    return m_raw->mName.C_Str();
}

void AssimpNode::setParent(AssimpNode* parent){
    m_parent = parent ;
}
void AssimpNode::setIndex(unsigned int index){
    m_index = index ; 
}
void AssimpNode::setDepth(unsigned int depth){
    m_depth = depth ; 
}

void AssimpNode::addChild(AssimpNode* child)
{
    m_children.push_back(child); 
}



AssimpNode* AssimpNode::getParent(){
    return m_parent ;  
}
unsigned int AssimpNode::getIndex(){
    return m_index ; 
}
unsigned int AssimpNode::getDepth(){
    return m_depth ; 
}
aiMatrix4x4 AssimpNode::getTransform(){
   return m_transform ; 
}
unsigned int AssimpNode::getNumChildren(){
    return m_children.size(); 
}
AssimpNode* AssimpNode::getChild(unsigned int n){
    return n < getNumChildren() ? m_children[n] : NULL ;
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
    unsigned int nprog = progeny();
    unsigned int nmesh = getNumMeshes() ;

    if(nchild > 8)
    {
        for(unsigned int i = 0; i < nmesh ; i++)
        {   
            unsigned int meshIndex = getMeshIndexRaw(i);

            aiMesh* rawmesh = getRawMesh(i);

            printf("AssimpNode::dump  i %d meshIndex %d \n", i, meshIndex);

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

     for(unsigned int i = 0; i < m_numMeshes ; i++)
     {
          aiMesh* src = getRawMesh(i);
          m_meshes[i] = new aiMesh ; 
          copyMesh(m_meshes[i], src, m_transform); 
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
    aiVector3D* nlow  = getLow();
    aiVector3D* nhigh = getHigh();

    if(m_low && m_high)
    {
        low.x = std::min( low.x, nlow->x);
        low.y = std::min( low.y, nlow->y);
        low.z = std::min( low.z, nlow->z);

        high.x = std::max( high.x, nhigh->x);
        high.y = std::max( high.y, nhigh->y);
        high.z = std::max( high.z, nhigh->z);
   } 
}


aiVector3D* AssimpNode::getLow()
{
    return m_low ; 
}
aiVector3D* AssimpNode::getHigh()
{
    return m_high ; 
}
aiVector3D* AssimpNode::getCenter()
{
    return m_center ; 
}
aiVector3D* AssimpNode::getExtent()
{
    return m_extent ; 
}




unsigned int AssimpNode::getNumMeshes()
{
    return m_numMeshes ; 
}
unsigned int AssimpNode::getNumMeshesRaw()
{
    return m_raw->mNumMeshes ; 
}


unsigned int AssimpNode::getMeshIndexRaw(unsigned int index)
{
    // node index to "global" scene mesh index
    return m_raw->mMeshes[index];
}

aiMesh* AssimpNode::getRawMesh(unsigned int index)
{
     unsigned int meshIndex = getMeshIndexRaw(index);
     aiMesh* mesh = m_tree->getRawMesh(meshIndex);
     return mesh ;
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
       printf("AssimpNode::ancestors %2d %s \n", count, node->getName() );
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






