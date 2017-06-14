oyoctogl-src(){      echo externals/oyoctogl.bash ; }
oyoctogl-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(oyoctogl-src)} ; }
oyoctogl-vi(){       vi $(oyoctogl-source) ; }
oyoctogl-env(){      olocal- ; opticks- ; }
oyoctogl-usage(){ cat << EOU

Yocto-GL as Opticks External
====================================

See also env-;yoctogl-

::

    y(){ opticks- ; oyoctogl- ; oyoctogl-cd ; git status ; }


* https://github.com/simoncblyth/yocto-gl/commits/master



CMake install locations
-------------------------

This has same CMake prefix as Opticks, thus the CMakeLists.txt
does some internal prefixing of install locations

::

    install(TARGETS ${name}  DESTINATION     externals/lib)
    install(FILES ${HEADERS} DESTINATION     externals/include/${name})
    install(FILES ${EXT_HEADERS} DESTINATION externals/include/${name}/ext)




Flattened glTF
-----------------


fl_scene : name and index refs, fl_gltf : the referenced fl_objects 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

yocto_gltf.h::

    1174 struct fl_scene {
    1176     std::string name = "";
    1178     std::vector<int> cameras;
    1180     std::vector<int> materials;
    1182     std::vector<int> textures;
    1184     std::vector<int> primitives;
    1186     std::vector<int> meshes;
    1188     std::vector<int> transforms;   // huh: reffing what ?
    1189 };

    1194 struct fl_gltf {
    1196     int default_scene = -1;
    1198     std::vector<fl_camera*> cameras;
    1200     std::vector<fl_material*> materials;
    1202     std::vector<fl_texture*> textures;
    1204     std::vector<fl_primitives*> primitives;
    1206     std::vector<fl_mesh*> meshes;
    1208     std::vector<fl_scene*> scenes;
    1209 };
    1210 

    1161 struct fl_mesh {
    1163     std::string name = "";
    1165     std::array<float, 16> xform = {
    1166         1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    1168     std::vector<int> primitives;
    1169 };
    1170 


yocto_gltf.cpp::

    2907 // Flattens a gltf file into a flattened asset.
    2909 YGLTF_API fl_gltf* flatten_gltf(const glTF_t* gltf, int scene_idx) {
    2911     auto fl_gltf = std::unique_ptr<ygltf::fl_gltf>(new ygltf::fl_gltf());

    2962     // convert meshes
    2963     auto meshes = std::vector<std::vector<int>>();      // vector-of-vectors of primitive indices

    2964     for (auto tf_mesh_ : gltf->meshes) {
    2965         auto tf_mesh = &tf_mesh_;
    2966         meshes.push_back({});
    2967         auto mesh = &meshes.back();
    2968         // primitives
    2969         for (auto& tf_primitives : tf_mesh->primitives) {
    2970             auto prim = new fl_primitives();
    2971             mesh->push_back((int)fl_gltf->primitives.size());
    ....
    3108             fl_gltf->primitives.push_back(prim);
    3109         }           
    3110     }           

    .... 
    3129     // walk the scenes and add objects
    3130     for (auto scn_id : scenes) {
    3131         auto scn = &gltf->scenes.at(scn_id);
    3132         auto fl_scn = new fl_scene();
    3133         auto stack = std::vector<std::tuple<int, std::array<float, 16>>>();
    3134         for (auto node_id : scn->nodes) {
    3135             stack.push_back(std::make_tuple(node_id, _identity_float4x4));
    3136         }
    3137         while (!stack.empty()) {
    3138             int node_id;
    3139             std::array<float, 16> xf;
    3140             std::tie(node_id, xf) = stack.back();
    3141             stack.pop_back();
    3142             auto node = &gltf->nodes.at(node_id);
    3143             xf = _float4x4_mul(xf, node_transform(node));

    ....
    3151             if (node->mesh >= 0) {
    3152 // BUG: initialization
    3153 #ifdef _WIN32
    3154                 auto fm = new fl_mesh();
    3155                 fm->name = gltf->meshes.at(mesh_name).name;
    3156                 fm->xform = xf;
    3157                 fm->primitives = meshes.at(mesh_name);
    3158                 fl_gltf->meshes.push_back(fm);
    3159 #else
    3160                 fl_gltf->meshes.push_back(
    3161                     new fl_mesh{gltf->meshes.at(node->mesh).name, xf,
    3162                         meshes.at(node->mesh)});
    3163 #endif
    3164                 fl_scn->meshes.push_back((int)fl_gltf->meshes.size() - 1);
    3165             }
    3166             for (auto child : node->children) { stack.push_back({child, xf}); }
    3167             fl_gltf->scenes.push_back(fl_scn);
    3168         }
    3169     }
    3170 
    3171     return fl_gltf.release();   // pass ownership to caller
    3172 }





EOU
}

oyoctogl-edit(){ vi $(opticks-home)/cmake/Modules/FindYoctoGL.cmake ; }
oyoctogl-url(){ echo https://github.com/simoncblyth/yocto-gl ; }


oyoctogl-dir(){  echo $(opticks-prefix)/externals/yoctogl/yocto-gl ; }
oyoctogl-bdir(){ echo $(opticks-prefix)/externals/yoctogl/yocto-gl.build ; }



oyoctogl-cd(){  cd $(oyoctogl-dir); }
oyoctogl-bcd(){ cd $(oyoctogl-bdir) ; }

oyoctogl-fullwipe()
{
    rm -rf $(opticks-prefix)/externals/yoctogl 
}

oyoctogl-update()
{
    oyoctogl-fullwipe
    oyoctogl-- 
}


oyoctogl-get(){
   local iwd=$PWD
   local dir=$(dirname $(oyoctogl-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d yocto-gl ] && git clone $(oyoctogl-url)
   cd $iwd
}

oyoctogl-cmake()
{
    local iwd=$PWD
    local bdir=$(oyoctogl-bdir)
    local sdir=$(oyoctogl-dir)

    #rm -rf $bdir
    mkdir -p $bdir
    #[ -f "$bdir/CMakeCache.txt" ] && echo $msg configured already && return  
    rm -f "$bdir/CMakeCache.txt"

    oyoctogl-bcd   
    opticks-


    cmake \
       -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
       $* \
       $sdir


    cd $iwd
}

oyoctogl-make()
{
    local iwd=$PWD
    oyoctogl-bcd
    cmake --build . --config Release --target ${1:-install}
    cd $iwd
}


oyoctogl--()
{
   oyoctogl-get
   oyoctogl-cmake
   oyoctogl-make install
}

oyoctogl-t()
{
   # oyoctogl-make test
   ygltf_reader $TMP/nd/scene.gltf
}





