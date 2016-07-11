// npy-
#include "NPY.hpp"
#include "MultiViewNPY.hpp"
#include "NGLM.hpp"

// okc-
#include "Composition.hh"
#include "Opticks.hh"

// oglrap-
#include "Interactor.hh"
#include "Frame.hh"
#include "Rdr.hh"
#include "Scene.hh"

#include "OGLRAP_LOG.hh"
#include "PLOG.hh"


class AxisTest {
  public:
      AxisTest(int argc, char** argv)
        :
         m_opticks(NULL),
         m_composition(NULL),
         m_scene(NULL), 
         m_frame(NULL),
         m_interactor(NULL),
         m_window(NULL),
         m_axis_renderer(NULL)
      {
         init(argc, argv);
      }

  private:
      void init(int argc, char** argv); 
      void initViz();
      void prepareViz();
      void upload();
      void render(); 
  public:
      void renderLoop();
  private:
      Opticks*     m_opticks ;
      Composition* m_composition ;
      Scene*       m_scene ;
      Frame*       m_frame ;
      Interactor*  m_interactor ;
      GLFWwindow*  m_window ;
      Rdr*         m_axis_renderer ; 
};


void AxisTest::init(int argc, char** argv)
{
    m_opticks = new Opticks(argc, argv);
    m_opticks->configure();

    initViz();
    prepareViz();
    upload();
}

void AxisTest::initViz()
{
    m_composition = new Composition ; 
    m_scene = new Scene ; 
    m_frame = new Frame ; 
    m_interactor = new Interactor ; 

    m_interactor->setFrame(m_frame);
    m_interactor->setScene(m_scene);
    m_interactor->setComposition(m_composition);
    
    m_scene->setInteractor(m_interactor);

    m_frame->setInteractor(m_interactor);
    m_frame->setComposition(m_composition);
    m_frame->setScene(m_scene);

    m_frame->setTitle("AxisTest");
}

void AxisTest::prepareViz()
{
    glm::uvec4 size = m_opticks->getSize();
    glm::uvec4 position = m_opticks->getPosition() ;

    m_composition->setSize( size );
    m_composition->setFramePosition( position );

    m_scene->setRenderMode("+axis"); 

    LOG(info) << "AxisTest::prepareViz initRenderers " ; 

    m_scene->initRenderers(); 

   // defer until renderers are setup, as distributes to them 
    m_scene->setComposition(m_composition);     
    LOG(info) << "AxisTest::prepareViz initRenderers DONE " ; 
    m_frame->init();  
    LOG(info) << "AxisTest::prepareViz frame init DONE " ; 
    m_window = m_frame->getWindow();

    LOG(info) << "AxisTest::prepareViz DONE " ; 
}


void AxisTest::upload()
{
    LOG(info) << "AxisTest::upload " ; 

    m_composition->update();
    m_axis_renderer = m_scene->getAxisRenderer();

    MultiViewNPY* axis_attr = m_composition->getAxisAttr(); 
    bool debug = true ; 
    m_axis_renderer->upload(axis_attr, debug ); 

    m_scene->setTarget(0, true);
    LOG(info) << "AxisTest::upload DONE " ; 
}

void AxisTest::render()
{
    m_frame->viewport();
    m_frame->clear();

    m_scene->render();
}

void AxisTest::renderLoop()
{
    m_frame->hintVisible(true);
    m_frame->show();
    LOG(info) << "after frame.show() "; 

    unsigned int count ; 

    while (!glfwWindowShouldClose(m_window))
    {    
        m_frame->listen(); 

        count = m_composition->tick();

        if( m_composition->hasChanged() || m_interactor->hasChanged() || count == 1)   
        {    
            render();

            glfwSwapBuffers(m_window);

            m_interactor->setChanged(false);  
            m_composition->setChanged(false);   // sets camera, view, trackball dirty status 
        }    
    }    
}


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    OGLRAP_LOG__ ; 
    LOG(info) << argv[0] ; 

    AxisTest ax(argc, argv); 
    ax.renderLoop();

    return 0 ; 
}

