/*

    ggv --loader 
    ggv --loader --idyb
    ggv --dbg --loader --idyb
    ggv --loader --jpmt

*/

#include "Types.hpp"
#include "GCache.hh"
#include "GLoader.hh"
#include "GGeo.hh"
#include "AssimpGGeo.hh"

int main(int argc, char* argv[])
{
    Types* m_types = new Types ;
    m_types->readFlags("$ENV_HOME/graphics/optixrap/cu/photon.h");

    GCache* m_cache = new GCache("GGEOVIEW_");

    GLoader* m_loader = new GLoader ;

    m_loader->setTypes(m_types);
    m_loader->setCache(m_cache);
    m_loader->setImp(&AssimpGGeo::load);    // setting GLoaderImpFunctionPtr

    bool nogeocache = false ; 
    m_loader->load(nogeocache);

    GGeo* m_ggeo = m_loader->getGGeo();
    m_ggeo->dumpTree();


    return 0 ;
}


/*


Jump into geocache for 2 volume geometry::

    delta:ggeoview blyth$ cd $(ggv --idyb --idp)

Check mergedmesh 0::

    In [1]: n = np.load("GMergedMesh/0/nodeinfo.npy")

    In [2]: n[n[:,0]>0]
    Out[2]: 
    array([[ 288,  483, 3158, 3157],
           [ 288,  617, 3159, 3158]], dtype=uint32)

    
    In [11]: i[i[:,0]>0][:,0].sum()      # sum nface
    Out[11]: 576

    In [12]: i[i[:,0]>0][:,1].sum()      # sum nvert
    Out[12]: 1100

Looks like a face transition in the indices::

    In [13]: f = np.load("GMergedMesh/0/indices.npy")

    In [28]:  f.reshape(-1,3)[278:288]
    Out[28]: 
    array([[470, 471, 472],
           [472, 471, 473],
           [473, 471, 474],
           [474, 475, 476],
           [476, 475, 477],
           [477, 475, 478],
           [478, 475, 479],
           [479, 480, 481],
           [481, 453, 482],
           [482, 453, 452]], dtype=int32)

    In [29]:  f.reshape(-1,3)[288:298]
    Out[29]: 
    array([[483, 484, 485],
           [483, 485, 486],
           [487, 488, 489],
           [487, 489, 490],
           [491, 492, 493],
           [491, 493, 494],
           [495, 496, 497],
           [495, 497, 498],
           [499, 500, 501],
           [499, 501, 502]], dtype=int32)

    In [32]: f.reshape(-1,3)[-10:]
    Out[32]: 
    array([[1076, 1077, 1078],
           [1079, 1080, 1081],
           [1082, 1076, 1078],
           [1083, 1084, 1085],
           [1086, 1087, 1088],
           [1089, 1090, 1091],
           [1092, 1086, 1088],
           [1093, 1094, 1095],
           [1096, 1097, 1098],
           [1099, 1093, 1095]], dtype=int32)

    In [33]: f.max()
    Out[33]: 1099


Vertex transition::

    In [35]: v[473:483]
    Out[35]: 
    array([[ -19642.455, -799778.75 ,   -5564.95 ],
           [ -19568.709, -800180.562,   -5564.95 ],
           [ -18079.461, -799699.562,   -5564.95 ],
           [ -19393.465, -800549.625,   -5564.95 ],
           [ -19128.682, -800860.75 ,   -5564.95 ],
           [ -18792.389, -801092.75 ,   -5564.95 ],
           [ -18407.51 , -801229.812,   -5564.95 ],
           [ -18079.461, -799699.562,   -5564.95 ],
           [ -18000.281, -801262.562,   -5564.95 ],
           [ -17598.449, -801188.812,   -5564.95 ]], dtype=float32)

    In [36]: v[483:493]      ## suspicious round numbers in z 
    Out[36]: 
    array([[ -17237.533, -801000.875,   -5565.   ],
           [ -17237.533, -801000.875,   -8635.   ],
           [ -16929.389, -800738.625,   -8635.   ],
           [ -16929.389, -800738.625,   -5565.   ],
           [ -16929.389, -800738.625,   -5565.   ],
           [ -16929.389, -800738.625,   -8635.   ],
           [ -16699.623, -800405.562,   -8635.   ],
           [ -16699.623, -800405.562,   -5565.   ],
           [ -16699.623, -800405.562,   -5565.   ],
           [ -16699.623, -800405.562,   -8635.   ]], dtype=float32)



*/

