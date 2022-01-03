#include <algorithm>

#include "SCenterExtentFrame.hh"
#include "SCenterExtentFrameTest.hh"
#include "SPresent.hh"

template<typename T>
SCenterExtentFrameTest<T>::SCenterExtentFrameTest( const SCenterExtentFrame<T>& _cef )
    :
    cef(_cef),
    ce(_cef.ce)
{    
    const glm::tmat4x4<T>& scale = cef.scale ; 
    const glm::tmat4x4<T>& iscale = cef.iscale ; 
    const glm::tmat4x4<T>& translate = cef.translate ; 
    const glm::tmat4x4<T>& itranslate = cef.itranslate ; 
    const glm::tmat4x4<T>& rotate = cef.rotate ; 
    const glm::tmat4x4<T>& irotate = cef.irotate ; 

    _world2model["0"]    = irotate * iscale * itranslate ; 
    _model2world["0"]    = translate * scale * rotate ; 
    modes.push_back("0");   pref_modes.push_back("0") ;   

    _world2model["SRT"]  = iscale * irotate * itranslate ; modes.push_back("SRT"); pref_modes.push_back("SRT") ;
    _world2model["RST"]  = irotate * iscale * itranslate ; modes.push_back("RST"); pref_modes.push_back("RST") ;
    _world2model["RTS"]  = irotate * itranslate * iscale ; modes.push_back("RTS");
    _world2model["TRS"]  = itranslate * irotate * iscale ; modes.push_back("TRS");
    _world2model["TSR"]  = itranslate * iscale * irotate ; modes.push_back("TSR");
    _world2model["STR"]  = iscale * itranslate * irotate ; modes.push_back("STR");

    _model2world["SRT"]  = scale * rotate * translate ;  
    _model2world["RST"]  = rotate * scale * translate ;  
    _model2world["RTS"]  = rotate * translate * scale ;
    _model2world["TRS"]  = translate * rotate * scale ;
    _model2world["TSR"]  = translate * scale * rotate ;
    _model2world["STR"]  = scale * translate * rotate ; 
}

/**
Observations

* NB THIS IS WITH UNIFORM S
* order of S and R makes no difference
* order of T and R matters 

**/

template<typename T>
void SCenterExtentFrameTest<T>::dump(char m)
{
    cef.dump("SCenterExtentFrameTest::dump"); 

    const std::vector<std::string>& mds = m == 'P' ? pref_modes : modes ;  

    for(unsigned pass=0 ; pass < 2 ; pass++)
    { 
        std::cout << std::endl << std::endl ; 
        for(unsigned i=0 ; i < mds.size() ; i++)
        {
            const std::string& mode = mds[i] ; 
            std::string rmode(mode); 
            std::reverse(rmode.begin(), rmode.end());
            const char* w2m_mode = mode.c_str() ;
            const char* m2w_mode = rmode.c_str() ;
            switch(pass)
            {
               case 0: std::cout << SPresent( _world2model[w2m_mode],  " world2model", w2m_mode )   << std::endl ; break ; 
               case 1: std::cout << SPresent( _model2world[m2w_mode],  " model2world", m2w_mode )   << std::endl ; break ;  
            }
        }
    }
}

template<typename T>
void SCenterExtentFrameTest<T>::check(const std::vector<glm::vec4>& world, const std::vector<std::string>& label, const char* title, const char* w2m_mode, const char* m2w_mode )
{
    const glm::tmat4x4<T>& w2m = _world2model[w2m_mode] ; 
    const glm::tmat4x4<T>& m2w = _model2world[m2w_mode] ;

    std::cout << std::endl << title << " w2m_mode " << w2m_mode << " m2w_mode " << m2w_mode << std::endl ; 
    for(unsigned i=0 ; i < world.size() ; i++)
    {
        const glm::vec4& world_pos = world[i] ; 
        glm::vec4 model_pos = w2m * world_pos ; 
        glm::vec4 world_pos2 = m2w * model_pos ; 

        std::cout 
            << std::setw(10) << label[i] 
            << SPresent( world_pos, "world_pos") 
            << SPresent( model_pos, "model_pos") 
            << SPresent( world_pos2, "world_pos2") 
            << std::endl
            ;
    }
}

template<typename T>
void SCenterExtentFrameTest<T>::check(char m)
{
    dump(); 
    std::vector<glm::vec4> world ;
    std::vector<std::string> label ; 

    world.push_back( { ce.x       , ce.y        , ce.z       , 1.0 } );  label.push_back("origin") ;  
    world.push_back( { ce.x + ce.w, ce.y        , ce.z       , 1.0 } );  label.push_back("+X") ; 
    world.push_back( { ce.x - ce.w, ce.y        , ce.z       , 1.0 } );  label.push_back("-X") ; 
    world.push_back( { ce.x       , ce.y+ce.w   , ce.z       , 1.0 } );  label.push_back("+Y") ; 
    world.push_back( { ce.x       , ce.y-ce.w   , ce.z       , 1.0 } );  label.push_back("-Y") ; 
    world.push_back( { ce.x       , ce.y        , ce.z+ce.w  , 1.0 } );  label.push_back("+Z") ; 
    world.push_back( { ce.x       , ce.y        , ce.z-ce.w  , 1.0 } );  label.push_back("-Z") ; 
    world.push_back( { ce.x + ce.w, ce.y + ce.w , ce.z       , 1.0 } );  label.push_back("+X+Y") ;  
    world.push_back( { ce.x - ce.w, ce.y - ce.w , ce.z       , 1.0 } );  label.push_back("-X-Y") ;  
    world.push_back( { ce.x - ce.w, ce.y - ce.w , ce.z - ce.w, 1.0 } );  label.push_back("-X-Y-Z") ; 
    world.push_back( { ce.x + ce.w, ce.y + ce.w , ce.z + ce.w, 1.0 } );  label.push_back("+X+Y+Z") ; 

    const std::vector<std::string>& mds = m == 'P' ? pref_modes : modes ;  

    for(unsigned i=0 ; i < mds.size() ; i++)
    {
        const std::string& mode = mds[i] ;  
        std::string rmode(mode); 
        std::reverse(rmode.begin(), rmode.end());

        const char* w2m_mode = mode.c_str() ;
        const char* m2w_mode = rmode.c_str() ;

        check(world, label, " world->model->world ", w2m_mode, m2w_mode ); 
    }
}

int main(int argc, char** argv)
{
    double cx = 100. ; 
    double cy = 100. ; 
    double cz = 100. ; 
    double extent = 5. ; 
    bool rtp_tangential = true ; 

    SCenterExtentFrame<double> f(cx,cy,cz,extent,rtp_tangential); 
    SCenterExtentFrameTest<double> ft(f);
    ft.check('P');  

    return 0 ; 
}

