
#include <assimp/cimport.h>        // Plain-C interface
#include <assimp/scene.h>          // Output data structure
#include <assimp/postprocess.h>    // Post processing flags


bool DoTheImportThing( const char* pFile)
{
  // Start the import on the given file with some example postprocessing
  // Usually - if speed is not the most important aspect for you - you'll t
  // probably to request more postprocessing than we do in this example.
  const aiScene* scene = aiImportFile( pFile, 
    aiProcess_CalcTangentSpace       | 
        aiProcess_Triangulate            |
        aiProcess_JoinIdenticalVertices  |
        aiProcess_SortByPType);
  // If the import failed, report it
  if( !scene)
  {
    //DoTheErrorLogging( aiGetErrorString());
    return false;
  }
  // Now we can access the file's contents
  //DoTheSceneProcessing( scene);

  // We're done. Release all resources associated with this import
  aiReleaseImport( scene);
  return true;
}



int main(int argc, char** argv)
{
   return 0 ;
}
