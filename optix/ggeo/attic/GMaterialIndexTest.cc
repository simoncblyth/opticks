#define GUI_ 1
#include "GMaterialIndex.hh"
#include "GColors.hh"
#include "GColorMap.hh"
#include "GBuffer.hh"

// npy-
#include "Counts.hpp"
#include "Index.hpp"
#include "jsonutil.hpp"

#include <iostream>
#include <iomanip>
#include "stdio.h"
#include "stdlib.h"




#include <boost/algorithm/string.hpp>

const char* custom_ordering =
             "GdDopedLS "
    "LiquidScintillator "
               "Acrylic "
            "MineralOil "
              "Bialkali "
              "IwsWater "
                 "Water "
             "DeadWater "
              "OwsWater "
                   "ESR "
    "UnstStainlessSteel "
        "StainlessSteel "
                "Vacuum "
                 "Pyrex "
                   "Air "
                  "Rock "
                   "PPE "
             "Aluminium "
 "ADTableStainlessSteel "
                  "Foam "
              "Nitrogen "
           "NitrogenGas "
                 "Nylon "
                   "PVC "
                 "Tyvek"
                       ;


void custom_material_indices()
{
    typedef std::vector<std::string> VS ;
    VS custom; 
    boost::split(custom, custom_ordering, boost::is_any_of(" ")); 

    Index idx("GMaterialIndex");
    idx.add(custom);
    idx.dump();

    const char* customdir = "$HOME/.opticks" ;
    idx.save(customdir);

    assert(true == existsPath(customdir, "GMaterialIndexSource.json")); 
    assert(true == existsPath(customdir, "GMaterialIndexLocal.json")); 
}




int main(int argc, char** argv)
{
    const char* idpath = getenv("IDPATH");

    custom_material_indices();

    GMaterialIndex* idx = GMaterialIndex::load(idpath);                      // itemname => index

    GColorMap* cmap = GColorMap::load(idpath, "GMaterialIndexColors.json");  // itemname => colorname 
    idx->setColorMap(cmap);

    GColors* colors = GColors::load(idpath,"GColors.json");           // colorname => hexcode 
    idx->setColorSource(colors);

    idx->test();
    idx->dump();

    //GBuffer* buffer = idx->getColorBuffer();
    //printf("makeColorBuffer %u \n", buffer->getNumBytes() );
    //colors->dump_uchar4_buffer(buffer);

    idx->formTable();
    //idx->gui();

    return 0 ;
}
