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

#pragma once

#include <vector>
#include <string>


class Opticks ; 
//class OpticksHub ; 

class Composition ;
class Bookmarks ; 

class Camera ; 
class View ; 
class Trackball ;
class Touchable ; 
class Frame ;   
class Scene ;   
class Animator ; 

#include "plog/Severity.h"


#include "OGLRAP_API_EXPORT.hh"
class OGLRAP_API Interactor {

       static const plog::Severity LEVEL ; 
  public:
       enum { NUM_KEYS = 512 } ;

       unsigned int getModifiers();

       static const char* keys ; 
       static const char* DRAGFACTOR ; 
       static const char* OPTIXMODE ; 

       Interactor(Composition* composition); 

       void gui();


       Bookmarks* getBookmarks();
       void setTouchable(Touchable* touchable);
       void setBookmarks( Bookmarks* bookmarks);
       void setScene(Scene* scene);
       void setFrame(Frame* frame);
       void setContainer(unsigned int container);

  private:
       //void setComposition(Composition* composition);
  public:
       bool isOptiXMode();
       void setOptiXMode(int optix_mode);
       int  getOptiXMode();
       void setOptiXResolutionScale(unsigned int scale);
       void nextOptiXResolutionScale(unsigned int modifiers);
       unsigned int getOptiXResolutionScale();
  public:
       void nextRenderStyle(unsigned modifiers);
  public:

       Touchable*   getTouchable();
       Frame*       getFrame();
       bool*        getScrubModeAddress();
       bool*        getGUIModeAddress();
       bool*        getLabelModeAddress();

       //bool*        getModeAddress(const char* name);
       unsigned int getContainer();
       bool         hasChanged();
       void         setChanged(bool changed);

  public:
       void cursor_drag( float x, float y, float dx, float dy, int ix, int iy );
       void touch(int ix, int iy);

   public:
        typedef enum { NONE, SCRUB, LABEL, FULL, NUM_GUI_STYLE } GUIStyle_t ;  
        void nextGUIStyle();
        void applyGUIStyle();
  public:
       void pan_mode_key_pressed(unsigned int modifiers);
       void y_key_pressed(unsigned int modifiers);
       void z_key_pressed(unsigned int modifiers);
       void number_key_pressed(unsigned int number);
       void number_key_released(unsigned int number);
       void key_pressed(unsigned int key);
       void key_released(unsigned int key);
       void space_pressed();
       void tab_pressed();
       void menu_pressed();

  public:
       void configureF(const char* name, std::vector<float> values);
       void configureI(const char* name, std::vector<int> values);

  public:
       void Print(const char* msg);
       void updateStatus();
       const char* getStatus();  // this feeds into Frame title, visible on window surround 

  private:
       //OpticksHub*  m_hub ; 
       Composition* m_composition ; 
       Bookmarks*   m_bookmarks ; 
       Camera*      m_camera ; 
       View*        m_view ; 
       Trackball*   m_trackball ; 
       Touchable*   m_touchable ; 
       Frame*       m_frame; 
       Scene*       m_scene; 
       Animator*    m_animator ; 
      

       bool m_zoom_mode ;
       bool m_pan_mode ;
       static const unsigned int _pan_mode_key ;
       bool m_near_mode ;
       bool m_far_mode ;
       bool m_yfov_mode ;
       bool m_scale_mode ;
       bool m_rotate_mode ;
       bool m_bookmark_mode ;
       bool m_gui_mode ;
       bool m_scrub_mode ;
       bool m_time_mode ;
       bool m_label_mode ;
       bool m_keys_down[NUM_KEYS] ; 

       unsigned int m_optix_resolution_scale ;

       float m_dragfactor ;
       unsigned int m_container ;

       bool  m_changed ; 

       enum { STATUS_SIZE = 128 };
       char m_status[STATUS_SIZE] ; 

       GUIStyle_t m_gui_style ; 

};



