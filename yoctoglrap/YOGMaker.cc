#include "YOGMaker.hh"

using ygltf::glTF_t ; 
using ygltf::node_t ; 
using ygltf::scene_t ; 
using ygltf::mesh_t ; 
using ygltf::mesh_primitive_t ; 
using ygltf::buffer_t ; 
using ygltf::buffer_data_t ; 
using ygltf::bufferView_t ; 
using ygltf::accessor_t ; 


ygltf::scene_t YOGMaker::make_scene(std::vector<int>& nodes)
{
    scene_t sc ;   
    sc.nodes = nodes ;
    return sc ; 
}
ygltf::node_t YOGMaker::make_node(int mesh, std::vector<int>& children)
{
    node_t no ; 
    no.mesh = mesh ; 
    no.children = children ; 
    return no ; 
}

ygltf::buffer_t YOGMaker::make_buffer(const char* uri,  ygltf::buffer_data_t& data )
{
    int byteLength = data.size();   // vector<unsigned char>
    buffer_t bu ; 
    bu.uri = uri ; 
    bu.byteLength = byteLength ; 
    bu.data = data ; 
    return bu ; 
}

ygltf::buffer_t YOGMaker::make_buffer(const char* uri,  int byteLength)
{
    buffer_t bu ; 
    bu.uri = uri ; 
    bu.byteLength = byteLength ; 
    return bu ; 
}
ygltf::bufferView_t YOGMaker::make_bufferView(
    int buffer, 
    int byteOffset, 
    int byteLength,  
    ygltf::bufferView_t::target_t target
)
{
    bufferView_t bv ; 
    bv.buffer = buffer ; 
    bv.byteOffset = byteOffset ; 
    bv.byteLength = byteLength ; 
    bv.target = target ; 
    return bv ; 
}
ygltf::accessor_t YOGMaker::make_accessor(
       int bufferView, 
       int byteOffset, 
       ygltf::accessor_t::componentType_t componentType, 
       int count,
       ygltf::accessor_t::type_t  type,
       std::vector<float>& min, 
       std::vector<float>& max 
)
{
     accessor_t ac ; 
     ac.bufferView = bufferView ; 
     ac.byteOffset = byteOffset ;
     ac.componentType = componentType ; 
     ac.count = count ; 
     ac.type = type ; 
     ac.min = min ; 
     ac.max = max ; 
     return ac ; 
} 

ygltf::mesh_primitive_t YOGMaker::make_mesh_primitive(
    std::map<std::string, int>& attributes,
    int indices, 
    int material, 
    ygltf::mesh_primitive_t::mode_t mode
)
{
    mesh_primitive_t mp ; 
    mp.attributes = attributes ; 
    mp.indices = indices ; 
    mp.material = material ; 
    mp.mode = mode ; 
    return mp ; 
}
ygltf::mesh_t YOGMaker::make_mesh( std::vector<ygltf::mesh_primitive_t> primitives )
{
    mesh_t mh ; 
    mh.primitives = primitives ; 
    return mh ; 
}

std::unique_ptr<glTF_t> YOGMaker::make_gltf_example()
{
    auto gltf = std::unique_ptr<glTF_t>(new glTF_t());

    std::vector<scene_t>& scenes = gltf->scenes ; 
    std::vector<node_t>& nodes = gltf->nodes ; 
    std::vector<buffer_t>& buffers = gltf->buffers ; 
    std::vector<bufferView_t>& bufferViews = gltf->bufferViews ; 
    std::vector<accessor_t>& accessors = gltf->accessors ; 
    std::vector<mesh_t>& meshes = gltf->meshes ; 

    int node = 0 ;   // index of root note 
    std::vector<int> scene_nodes = {node} ;  
    scene_t sc = make_scene(scene_nodes) ;
 

    std::vector<int> children = {} ;

    int mesh = 0 ;  // index of first mesh 
    node_t no = make_node(mesh, children) ; 


    int buffer = 0 ;  // index of first buffer
    buffer_t bu = make_buffer(
         "data:application/octet-stream;base64,AAABAAIAAAAAAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAAAAAACAPwAAAAA=", 
         44
    ); 

    enum {
      indices = 0, 
      vertices = 1,
      num = 2
    };

    bufferView_t bv[num] ; 
    accessor_t   ac[num] ; 

    int indices_byteOffset = 0 ; 
    int indices_byteLength = 6 ; 
    int vertices_byteOffset = 8 ; 
    int vertices_byteLength = 36 ; 

    bv[indices]  = make_bufferView(
          buffer, 
          indices_byteOffset, 
          indices_byteLength,  
          bufferView_t::target_t::element_array_buffer_t 
    ) ; 

    bv[vertices]  = make_bufferView(
          buffer, 
          vertices_byteOffset, 
          vertices_byteLength, 
          bufferView_t::target_t::array_buffer_t 
    ) ; 


    int count = 3 ; 

    std::vector<float> indices_min = { 0 } ; 
    std::vector<float> indices_max = { 2 } ; 

    ac[indices] = make_accessor( 
                          indices, 
                          0,           // byteOffset
                          accessor_t::componentType_t::unsigned_short_t, 
                          count, 
                          accessor_t::type_t::scalar_t,
                          indices_min,
                          indices_max 
                       ) ;   


    std::vector<float> vertices_min = { 0, 0, 0 } ; 
    std::vector<float> vertices_max = { 1, 1, 1 } ; 

    ac[vertices] = make_accessor( 
                          vertices, 
                          0,          // byteOffset
                          accessor_t::componentType_t::float_t, 
                          count, 
                          accessor_t::type_t::vec3_t,
                          vertices_min,
                          vertices_max 
                       ) ;   


    int indices_accessor = indices ; 
    int vertices_accessor = vertices ; 
    int material = -1 ; 
    std::map<std::string, int> attributes = {{"POSITION", vertices_accessor }} ; 

    mesh_primitive_t mp = make_mesh_primitive( 
                             attributes,
                             indices_accessor, 
                             material,
                             mesh_primitive_t::mode_t::triangles_t 
                         ) ; 

    std::vector<mesh_primitive_t> primitives ; 
    primitives = { mp } ;
   
    mesh_t mh = make_mesh( primitives ) ; 

    nodes = {no} ; 
    scenes = {sc} ;  
    buffers = {bu} ; 
    bufferViews = {bv[indices], bv[vertices]} ;  
    accessors = {ac[indices], ac[vertices]} ;
    meshes = {mh} ; 


    return gltf ; 
}


std::unique_ptr<glTF_t> YOGMaker::make_gltf()
{
    // NB : there is no checking can easily construct non-sensical gltf 
    //      as shown below 

    auto gltf = std::unique_ptr<glTF_t>(new glTF_t());
    std::vector<node_t>& nodes = gltf->nodes ; 
    std::vector<scene_t>& scenes = gltf->scenes ; 

    node_t a, b, c  ; 
    a.children = {2} ;   
    b.children = {3} ;   
    c.children = {} ;   

    nodes = { a, b, c } ;

    scene_t sc ;    // scene references nodes by index
    sc.nodes = {1,2,3} ;

    scenes = {sc} ;  

    return gltf ; 
}



/*

ygltf buffers use
    std::vector<unsigned char> 

hmm how to put NPY data arrays into ygltf buffers 

* https://docs.scipy.org/doc/numpy/neps/npy-format.html

Need a way to predict the byteLength that the NPY will 
occupy in a file and the byteOffset to the start of the data 


epsilon:GBndLib blyth$ xxd GBndLibIndex.npy
00000000: 934e 554d 5059 0100 4600 7b27 6465 7363  .NUMPY..F.{'desc
00000010: 7227 3a20 273c 7534 272c 2027 666f 7274  r': '<u4', 'fort
00000020: 7261 6e5f 6f72 6465 7227 3a20 4661 6c73  ran_order': Fals
00000030: 652c 2027 7368 6170 6527 3a20 2831 3233  e, 'shape': (123
00000040: 2c20 3429 2c20 7d20 2020 2020 2020 200a  , 4), }        .
00000050: 0c00 0000 ffff ffff ffff ffff 0c00 0000  ................
00000060: 0c00 0000 ffff ffff ffff ffff 0b00 0000  ................
00000070: 0b00 0000 ffff ffff ffff ffff 0e00 0000  ................:w


*/



