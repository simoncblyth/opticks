
#include "BStr.hh"
#include "BFile.hh"
#include "Opticks.hh"

#include "GMesh.hh"
#include "GMeshLib.hh"
#include "GItemIndex.hh"

#include "PLOG.hh"


const unsigned GMeshLib::MAX_MESH = 500 ; 
const char* GMeshLib::GMESHLIB_INDEX = "MeshIndex" ; 
const char* GMeshLib::GMESHLIB = "GMeshLib" ; 

GMeshLib::GMeshLib(Opticks* ok) 
   :
   m_ok(ok),
   m_meshindex(NULL),
   m_missing(std::numeric_limits<unsigned>::max())
{
}

GMeshLib* GMeshLib::load(Opticks* ok)
{
    GMeshLib* meshlib = new GMeshLib(ok);
    meshlib->loadFromCache(); 
    return meshlib ; 
}

void GMeshLib::loadFromCache()
{
    const char* idpath = m_ok->getIdPath() ;
    m_meshindex = GItemIndex::load(idpath, GMESHLIB_INDEX);

    loadMeshes(idpath);
}

void GMeshLib::save() const
{
    const char* idpath = m_ok->getIdPath() ;
    m_meshindex->save(idpath);

    saveMeshes(idpath);
}



GItemIndex* GMeshLib::getMeshIndex()
{
    return m_meshindex ; 
}
const char* GMeshLib::getMeshName(unsigned aindex) 
{
    return m_meshindex->getNameSource(aindex);
}
unsigned GMeshLib::getMeshIndex(const char* name, bool startswith) 
{
    unsigned aindex = startswith ? 
          m_meshindex->getIndexSourceStarting(name, m_missing)
          :
          m_meshindex->getIndexSource(name, m_missing)
          ;
 
    assert( aindex != m_missing );
    return aindex ; 
}

void GMeshLib::dump(const char* msg) const
{
    unsigned num_mesh = m_meshindex->getNumItems();
    LOG(info) << msg 
              << " num_mesh " << num_mesh 
              ; 

    //m_meshindex->dump();

    for(unsigned i=0 ; i < num_mesh ; i++)
    {
        const char* name = m_meshindex->getNameSource(i);  // hmm assumes source index is 0:N-1 which is not gauranteed
        //const char* name = m_meshindex->getNameLocal(i);

        assert(name);
        unsigned aindex = m_meshindex->getIndexSource(name, m_missing); // huh : why have to go via name
        assert(aindex != m_missing);
        assert(aindex == i);

        std::cout 
               << " aidx " << std::setw(3) << aindex  
               << " name " << std::setw(40) << name 
               ;

         GMesh* mesh = getMesh(aindex);
         if(mesh)
         {
             assert( strcmp(mesh->getName(), name) == 0);
             assert( mesh->getIndex() == aindex );
             std::cout << mesh->desc() << std::endl ; 
         }
         else
         {
             std::cout << " NO MESH "  << std::endl ; 
         }
    }
}










unsigned int GMeshLib::getNumMeshes() const 
{
    return m_meshes.size();
}

GMesh* GMeshLib::getMesh(unsigned aindex) const 
{
    GMesh* mesh = NULL ; 
    for(unsigned int i=0 ; i < m_meshes.size() ; i++ )
    { 
        if(m_meshes[i]->getIndex() == aindex )
        {
            mesh = m_meshes[i] ; 
            break ; 
        }
    }
    return mesh ;
}  


GMesh* GMeshLib::getMesh(const char* name, bool startswith)
{
    unsigned aindex = getMeshIndex(name, startswith);
    return getMesh(aindex);
}



void GMeshLib::add(GMesh* mesh)
{
    if(!m_meshindex) m_meshindex = new GItemIndex(GMESHLIB_INDEX)   ;

    m_meshes.push_back(mesh);

    const char* name = mesh->getName();
    unsigned int index = mesh->getIndex();

    LOG(info) << "GMeshLib::add (GMesh)"
              << " index " << std::setw(4) << index 
              << " name " << name 
              ;

    m_meshindex->add(name, index); 
}




void GMeshLib::removeMeshes(const char* idpath ) const 
{
   for(unsigned int idx=0 ; idx < MAX_MESH ; ++idx)
   {   
        const char* sidx = BStr::itoa(idx);
        if(BFile::ExistsDir(idpath, GMESHLIB, sidx))
        { 
            BFile::RemoveDir(idpath, GMESHLIB, sidx); 
        }
   } 
}




void GMeshLib::loadMeshes(const char* idpath )
{
   LOG(info) << "idpath "  << idpath ;  
   for(unsigned int idx=0 ; idx < MAX_MESH ; ++idx)
   {   
        const char* sidx = BStr::itoa(idx);
        std::string spath = BFile::FormPath(idpath, GMESHLIB, sidx);
        const char* path = spath.c_str() ;

        if(BFile::ExistsDir(path))
        {   
            LOG(debug) << "GMeshLib::loadMeshes " << path ; 
            GMesh* mesh = GMesh::load( path );

            const char* name = getMeshName(idx);
            assert(name);

            mesh->setIndex(idx);
            mesh->setName(strdup(name));

            m_meshes.push_back(mesh);
        }
        else
        {
            LOG(debug) << "GMeshLib::loadMeshes " 
                       << " no mdir for idx " << idx 
                       << " path " << path 
                       ;
        }
   }
   LOG(debug) << "GMeshLib::loadMeshes" 
             << " loaded "  << m_meshes.size()
             ;
}

void GMeshLib::saveMeshes(const char* idpath) const 
{
    removeMeshes(idpath); // clean old meshes to avoid duplication when repeat counts go down 

    typedef std::vector<GMesh*>::const_iterator VMI ; 
    for(VMI it=m_meshes.begin() ; it != m_meshes.end() ; it++)
    {
        GMesh* mesh = *it ; 
        unsigned int idx = mesh->getIndex() ; 
        const char* sidx = BStr::itoa(idx);

        mesh->save(idpath, GMESHLIB, sidx); 
    }
}


