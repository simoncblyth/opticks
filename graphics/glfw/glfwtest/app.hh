#ifndef APP_H
#define APP_H

class Scene ;

class App {
   public:
       App();
       virtual ~App();
       
       void setSize(unsigned int width, unsigned int height);
       void setTitle(const char* title);
       void setScene(Scene* scene);

       GLFWwindow* getWindow();

       void init();
       void runloop();
       void exit();

   private:
       void handle_event(GLEQevent& event);
       void dump_event(GLEQevent& event);
       void resize(unsigned int width, unsigned int height);
       void key_pressed(unsigned int key);
       void render();

   private:
       unsigned int m_width ; 
       unsigned int m_height ; 
       const char* m_title ;
       GLFWwindow* m_window;
       Scene* m_scene ;

};

#endif


