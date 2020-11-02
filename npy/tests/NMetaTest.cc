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

// TEST=NMetaTest om-t

#include <string>

#include "NMeta.hpp"

#include "OPTICKS_LOG.hh"

void test_composable()
{
    const char* path = "$TMP/NMetaTest/test_composable.json" ;

    NMeta m1 ; 
    assert( m1.size() == 0 ); 

    m1.set<int>("red", 1);
    assert( m1.size() == 1 ); 
    m1.set<int>("green", 2);
    m1.set<int>("blue", 3);
    m1.set<float>("pi", 3.1415);
    m1.set<std::string>("name", "yo");

    assert( m1.size() == 5 ); 
    assert( m1.getNumKeys() == 5 ); 

    NMeta m2 ; 
    m2.set<int>("cyan", 7);
    m2.set<int>("red", 100);
    m2.set<int>("green", 200);
    m2.set<int>("blue", 300);
    m2.set<float>("pi", 3.1415);
    m2.set<std::string>("name", "yo");

    assert( m2.getNumKeys() == 6 ); 

    NMeta m ;
    m.setObj("m1", &m1 );
    m.setObj("m2", &m2 );
    m.setObj("m3", &m2 );
    m.setObj("m4", &m2 );

    unsigned xkey = 4 ; 
    assert( m.getNumKeys() == xkey ); 

    m.dump();  
    m.save(path);

    NMeta* ml = NMeta::Load(path);
    ml->dump();

    unsigned num_key = ml->getNumKeys() ;
    assert( num_key == xkey );

    for(unsigned i=0 ; i < num_key ; i++)
    {
        const char* key = ml->getKey(i);
        NMeta* lm = ml->getObj(key);
        LOG(info) 
            << " i " << i 
            << " key " << key 
            << " lm " << lm->desc()
            ; 
    }
}

void test_write_read()
{
    const char* path = "$TMP/NMetaTest/test_write_read.json" ;

    NMeta m ; 
    m.set<int>("red", 1);
    m.set<int>("green", 2);
    m.set<int>("blue", 3);
    m.set<float>("pi", 3.1415);
    m.set<std::string>("name", "yo");

    m.save(path);
    m.dump();  
}

void test_copy_ctor()
{
    NMeta m ; 
    m.set<int>("red", 1);
    m.set<int>("green", 2);
    m.set<int>("blue", 3);
    m.set<float>("pi", 3.1415);
    m.set<std::string>("name", "yo");

    assert( m.getNumKeys() == 5 );

    m.dump();  

    NMeta mc(m);  
    mc.dump("copy-ctor");

    assert( mc.getNumKeys() == 5 );
}


void test_addEnvvarsWithPrefix()
{
    NMeta m ; 
    m.addEnvvarsWithPrefix("OPTICKS_");
    m.dump("addEnvvarsWithPrefix") ; 

    std::vector<std::string> lines = m.getLines(); 
    for(unsigned i=0 ; i < lines.size() ; i++) 
    {
        std::cout << lines[i] << std::endl ; 
    }
}

void test_appendString()
{

    NMeta m ; 
    m.set<std::string>("name", "yo");
    m.dump();  

    m.appendString("name","yo");
    m.appendString("name","yo");
    m.appendString("name","yo");
    m.appendString("name","yo");

    m.dump();  
}

void test_prepLines()
{

    NMeta m ; 
    m.set<std::string>("name", "yo");
    m.set<int>("cyan", 7);
    m.set<int>("red", 100);
    m.dump();  

    m.dumpLines();
}

void test_append()
{

    NMeta a ; 
    a.set<std::string>("name", "yo");
    a.set<int>("cyan", 7);
    a.set<int>("red", 100);
    a.dump();  

    NMeta b ; 
    b.set<std::string>("name", "yo2");
    b.set<int>("green", 7);
    b.set<int>("blue", 100);
    b.dump();  


    a.append(&b);
    a.dump();  
}


const char* COLORMAP_NAME2HEX = R"LITERAL(

   {"indigo": "#4B0082", "gold": "#FFD700", "hotpink": "#FF69B4", "firebrick": "#B22222", 
 "indianred": "#CD5C5C", "yellow": "#FFFF00", "mistyrose": "#FFE4E1", "darkolivegreen": "#556B2F", 
 "olive": "#808000", "darkseagreen": "#8FBC8F", "pink": "#FFC0CB", "tomato": "#FF6347", "lightcoral": "#F08080", 
 "orangered": "#FF4500", "navajowhite": "#FFDEAD", "lime": "#00FF00", "palegreen": "#98FB98", 
 "darkslategrey": "#2F4F4F", "greenyellow": "#ADFF2F", "burlywood": "#DEB887", "seashell": "#FFF5EE", 
 "mediumspringgreen": "#00FA9A", "fuchsia": "#FF00FF", "papayawhip": "#FFEFD5", "blanchedalmond": "#FFEBCD", 
 "chartreuse": "#7FFF00", "dimgray": "#696969", "black": "#000000", "peachpuff": "#FFDAB9", "springgreen": "#00FF7F", 
 "aquamarine": "#7FFFD4", "white": "#FFFFFF", "orange": "#FFA500", "lightsalmon": "#FFA07A", "darkslategray": "#2F4F4F", 
 "brown": "#A52A2A", "ivory": "#FFFFF0", "dodgerblue": "#1E90FF", "peru": "#CD853F", "darkgrey": "#A9A9A9", 
 "lawngreen": "#7CFC00", "chocolate": "#D2691E", "crimson": "#DC143C", "forestgreen": "#228B22", "slateblue": "#6A5ACD", 
 "lightseagreen": "#20B2AA", "cyan": "#00FFFF", "mintcream": "#F5FFFA", "silver": "#C0C0C0", "antiquewhite": "#FAEBD7", 
 "mediumorchid": "#BA55D3", "skyblue": "#87CEEB", "gray": "#808080", "darkturquoise": "#00CED1", "goldenrod": "#DAA520", 
 "darkgreen": "#006400", "floralwhite": "#FFFAF0", "darkviolet": "#9400D3", "darkgray": "#A9A9A9", "moccasin": "#FFE4B5", 
 "saddlebrown": "#8B4513", "grey": "#808080", "darkslateblue": "#483D8B", "lightskyblue": "#87CEFA", "lightpink": "#FFB6C1", 
 "mediumvioletred": "#C71585", "slategrey": "#708090", "red": "#FF0000", "deeppink": "#FF1493", "limegreen": "#32CD32", 
 "darkmagenta": "#8B008B", "palegoldenrod": "#EEE8AA", "plum": "#DDA0DD", "turquoise": "#40E0D0", "lightgrey": "#D3D3D3", 
 "lightgoldenrodyellow": "#FAFAD2", "darkgoldenrod": "#B8860B", "lavender": "#E6E6FA", "maroon": "#800000", "yellowgreen": "#9ACD32", 
 "sandybrown": "#FAA460", "thistle": "#D8BFD8", "violet": "#EE82EE", "navy": "#000080", "magenta": "#FF00FF", 
 "dimgrey": "#696969", "tan": "#D2B48C", "rosybrown": "#BC8F8F", "olivedrab": "#6B8E23", "blue": "#0000FF", 
 "lightblue": "#ADD8E6", "ghostwhite": "#F8F8FF", "honeydew": "#F0FFF0", "cornflowerblue": "#6495ED", "linen": "#FAF0E6", 
 "darkblue": "#00008B", "powderblue": "#B0E0E6", "seagreen": "#2E8B57", "darkkhaki": "#BDB76B", "snow": "#FFFAFA", 
 "sienna": "#A0522D", "mediumblue": "#0000CD", "royalblue": "#4169E1", "lightcyan": "#E0FFFF", "green": "#008000", 
 "mediumpurple": "#9370DB", "midnightblue": "#191970", "cornsilk": "#FFF8DC", "paleturquoise": "#AFEEEE", 
 "bisque": "#FFE4C4", "slategray": "#708090", "darkcyan": "#008B8B", "khaki": "#F0E68C", "wheat": "#F5DEB3", 
 "teal": "#008080", "darkorchid": "#9932CC", "deepskyblue": "#00BFFF", "salmon": "#FA8072", "darkred": "#8B0000", 
 "steelblue": "#4682B4", "palevioletred": "#DB7093", "lightslategray": "#778899", "aliceblue": "#F0F8FF", "lightslategrey": "#778899", 
 "lightgreen": "#90EE90", "orchid": "#DA70D6", "gainsboro": "#DCDCDC", "mediumseagreen": "#3CB371", "lightgray": "#D3D3D3", 
 "mediumturquoise": "#48D1CC", "lemonchiffon": "#FFFACD", "cadetblue": "#5F9EA0", "lightyellow": "#FFFFE0", "lavenderblush": "#FFF0F5", 
 "coral": "#FF7F50", "purple": "#800080", "aqua": "#00FFFF", "whitesmoke": "#F5F5F5", "mediumslateblue": "#7B68EE", 
 "darkorange": "#FF8C00", "mediumaquamarine": "#66CDAA", "darksalmon": "#E9967A", "beige": "#F5F5DC", "blueviolet": "#8A2BE2", 
 "azure": "#F0FFFF", "lightsteelblue": "#B0C4DE", "oldlace": "#FDF5E6"}

)LITERAL";


void test_readTxt()
{
    NMeta* a = NMeta::FromTxt(COLORMAP_NAME2HEX); 
    a->dump();
    LOG(info) << " green is " << a->get<std::string>("green") ; 
}


void test_fillMap()
{
    NMeta* a = NMeta::FromTxt(COLORMAP_NAME2HEX); 

    typedef std::map<std::string, std::string> MSS ; 

    MSS m ; 
    a->fillMap(m); 

    LOG(info) << m.size() ; 

    for(MSS::const_iterator it=m.begin() ; it != m.end() ; it++)
    {
        std::cout 
           << std::setw(20) << it->first 
           << " : " 
           << std::setw(20) << it->second
           << std::endl 
           ;
    }
}


void test_add_string_NULL()
{
    NMeta* p = new NMeta ; 
    p->add<std::string>("HOSTNAME", "yo" );

    // p->add<std::string>("WHAT_ABOUT_NULL", NULL );
    // std::string s = NULL ;
    // std::string s(NULL) ;

/**

terminate called after throwing an instance of 'std::logic_error'
  what():  basic_string::_S_construct null not valid
Aborted (core dumped)

**/

     p->dump(); 
}


void test_hasKey()
{

    NMeta m1 ; 
    assert( m1.size() == 0 ); 

    m1.set<int>("red", 1);
    assert( m1.size() == 1 ); 
    m1.set<int>("green", 2);
    m1.set<int>("blue", 3);
    m1.set<float>("pi", 3.1415);
    m1.set<std::string>("name", "yo");

    assert( m1.size() == 5 ); 
    assert( m1.getNumKeys() == 5 ); 

    NMeta m2 ; 
    m2.set<int>("cyan", 7);
    m2.set<int>("red", 100);
    m2.set<int>("green", 200);
    m2.set<int>("blue", 300);
    m2.set<float>("pi", 3.1415);
    m2.set<std::string>("name", "yo");

    assert( m2.getNumKeys() == 6 ); 

    NMeta m ;
    m.setObj("m1", &m1 );
    m.setObj("m2", &m2 );
    m.setObj("m3", &m2 );
    m.setObj("m4", &m2 );

    assert( m.hasKey("m1") == true ); 
    assert( m.hasKey("m5") == false ); 
    assert( m.hasKey("m5") == false ); 

}




int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);
/*
    test_write_read();
    test_composable();
    test_copy_ctor();
    test_addEnvvarsWithPrefix();  
    test_appendString();  
    test_prepLines();  
    test_append();  
    test_readTxt();  
    test_fillMap();  
    test_add_string_NULL(); 
*/
    test_hasKey(); 


    return 0 ; 
}
// om-;TEST=NMetaTest om-t
