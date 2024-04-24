#pragma once

#include "gleq.h"

struct SGLFW_GLEQ
{
    static constexpr const char* GLEQ_NONE_                  = "GLEQ_NONE" ; 
    static constexpr const char* GLEQ_WINDOW_MOVED_          = "GLEQ_WINDOW_MOVED" ; 
    static constexpr const char* GLEQ_WINDOW_RESIZED_        = "GLEQ_WINDOW_RESIZED" ; 
    static constexpr const char* GLEQ_WINDOW_CLOSED_         = "GLEQ_WINDOW_CLOSED" ;
    static constexpr const char* GLEQ_WINDOW_REFRESH_        = "GLEQ_WINDOW_REFRESH" ; 
    static constexpr const char* GLEQ_WINDOW_FOCUSED_        = "GLEQ_WINDOW_FOCUSED" ;
    static constexpr const char* GLEQ_WINDOW_DEFOCUSED_      = "GLEQ_WINDOW_DEFOCUSED" ; 
    static constexpr const char* GLEQ_WINDOW_ICONIFIED_      = "GLEQ_WINDOW_ICONIFIED" ; 
    static constexpr const char* GLEQ_WINDOW_UNICONIFIED_    = "GLEQ_WINDOW_UNICONIFIED" ; 
    static constexpr const char* GLEQ_FRAMEBUFFER_RESIZED_   = "GLEQ_FRAMEBUFFER_RESIZED" ; 
    static constexpr const char* GLEQ_BUTTON_PRESSED_        = "GLEQ_BUTTON_PRESSED" ; 
    static constexpr const char* GLEQ_BUTTON_RELEASED_       = "GLEQ_BUTTON_RELEASED" ; 
    static constexpr const char* GLEQ_CURSOR_MOVED_          = "GLEQ_CURSOR_MOVED" ; 
    static constexpr const char* GLEQ_CURSOR_ENTERED_        = "GLEQ_CURSOR_ENTERED" ; 
    static constexpr const char* GLEQ_CURSOR_LEFT_           = "GLEQ_CURSOR_LEFT" ; 
    static constexpr const char* GLEQ_SCROLLED_              = "GLEQ_SCROLLED" ; 
    static constexpr const char* GLEQ_KEY_PRESSED_           = "GLEQ_KEY_PRESSED" ; 
    static constexpr const char* GLEQ_KEY_REPEATED_          = "GLEQ_KEY_REPEATED" ; 
    static constexpr const char* GLEQ_KEY_RELEASED_          = "GLEQ_KEY_RELEASED" ; 
    static constexpr const char* GLEQ_CODEPOINT_INPUT_       = "GLEQ_CODEPOINT_INPUT" ; 
    static constexpr const char* GLEQ_MONITOR_CONNECTED_     = "GLEQ_MONITOR_CONNECTED" ; 
    static constexpr const char* GLEQ_MONITOR_DISCONNECTED_  = "GLEQ_MONITOR_DISCONNECTED" ; 
#if GLFW_VERSION_MINOR >= 1
    static constexpr const char* GLEQ_FILE_DROPPED_          = "GLEQ_FILE_DROPPED" ; 
#endif
#if GLFW_VERSION_MINOR >= 2
    static constexpr const char* GLEQ_JOYSTICK_CONNECTED_    = "GLEQ_JOYSTICK_CONNECTED" ; 
    static constexpr const char* GLEQ_JOYSTICK_DISCONNECTED_ = "GLEQ_JOYSTICK_DISCONNECTED" ; 
#endif
#if GLFW_VERSION_MINOR >= 3
    static constexpr const char* GLEQ_WINDOW_MAXIMIZED_      = "GLEQ_WINDOW_MAXIMIZED" ; 
    static constexpr const char* GLEQ_WINDOW_UNMAXIMIZED_    = "GLEQ_WINDOW_UNMAXIMIZED" ;  
    static constexpr const char* GLEQ_WINDOW_SCALE_CHANGED_  = "GLEQ_WINDOW_SCALE_CHANGED" ;  
#endif
    static constexpr const char* SGLFW_GLEQ_INVALID_ENUM_    = "SGLFW_GLEQ_INVALID_ENUM" ;  
    static const char* Name(GLEQtype type); 
};

inline const char* SGLFW_GLEQ::Name(GLEQtype type)
{
    const char* s = nullptr ; 
    switch(type)
    {
        case GLEQ_NONE:                  s = GLEQ_NONE_                   ; break ;
        case GLEQ_WINDOW_MOVED:          s = GLEQ_WINDOW_MOVED_           ; break ;
        case GLEQ_WINDOW_RESIZED:        s = GLEQ_WINDOW_RESIZED_         ; break ;
        case GLEQ_WINDOW_CLOSED:         s = GLEQ_WINDOW_CLOSED_          ; break ;
        case GLEQ_WINDOW_REFRESH:        s = GLEQ_WINDOW_REFRESH_         ; break ;
        case GLEQ_WINDOW_FOCUSED:        s = GLEQ_WINDOW_FOCUSED_         ; break ;
        case GLEQ_WINDOW_DEFOCUSED:      s = GLEQ_WINDOW_DEFOCUSED_       ; break ;
        case GLEQ_WINDOW_ICONIFIED:      s = GLEQ_WINDOW_ICONIFIED_       ; break ;
        case GLEQ_WINDOW_UNICONIFIED:    s = GLEQ_WINDOW_UNICONIFIED_     ; break ;
        case GLEQ_FRAMEBUFFER_RESIZED:   s = GLEQ_FRAMEBUFFER_RESIZED_    ; break ;
        case GLEQ_BUTTON_PRESSED:        s = GLEQ_BUTTON_PRESSED_         ; break ;
        case GLEQ_BUTTON_RELEASED:       s = GLEQ_BUTTON_RELEASED_        ; break ;
        case GLEQ_CURSOR_MOVED:          s = GLEQ_CURSOR_MOVED_           ; break ;
        case GLEQ_CURSOR_ENTERED:        s = GLEQ_CURSOR_ENTERED_         ; break ;
        case GLEQ_CURSOR_LEFT:           s = GLEQ_CURSOR_LEFT_            ; break ;
        case GLEQ_SCROLLED:              s = GLEQ_SCROLLED_               ; break ;
        case GLEQ_KEY_PRESSED:           s = GLEQ_KEY_PRESSED_            ; break ;
        case GLEQ_KEY_REPEATED:          s = GLEQ_KEY_REPEATED_           ; break ;
        case GLEQ_KEY_RELEASED:          s = GLEQ_KEY_RELEASED_           ; break ;
        case GLEQ_CODEPOINT_INPUT:       s = GLEQ_CODEPOINT_INPUT_        ; break ;
        case GLEQ_MONITOR_CONNECTED:     s = GLEQ_MONITOR_CONNECTED_      ; break ;
        case GLEQ_MONITOR_DISCONNECTED:  s = GLEQ_MONITOR_DISCONNECTED_   ; break ;
#if GLFW_VERSION_MINOR >= 1                                               
        case GLEQ_FILE_DROPPED:          s = GLEQ_FILE_DROPPED_           ; break ;
#endif
#if GLFW_VERSION_MINOR >= 2
        case GLEQ_JOYSTICK_CONNECTED:    s = GLEQ_JOYSTICK_CONNECTED_     ; break ;
        case GLEQ_JOYSTICK_DISCONNECTED: s = GLEQ_JOYSTICK_DISCONNECTED_  ; break ;
#endif
#if GLFW_VERSION_MINOR >= 3
        case GLEQ_WINDOW_MAXIMIZED:      s = GLEQ_WINDOW_MAXIMIZED_       ; break ;
        case GLEQ_WINDOW_UNMAXIMIZED:    s = GLEQ_WINDOW_UNMAXIMIZED_     ; break ;
        case GLEQ_WINDOW_SCALE_CHANGED:  s = GLEQ_WINDOW_SCALE_CHANGED_   ; break ;
#endif
        default:                         s = SGLFW_GLEQ_INVALID_ENUM_     ; break ;
    }
    return s ; 
}

