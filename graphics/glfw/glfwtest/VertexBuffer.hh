#ifndef VERTEXBUFFER_H
#define VERTEXBUFFER_H

class Array ;

class VertexBuffer {
   public:
      VertexBuffer( Array* vertices, Array* indices );
      virtual ~VertexBuffer();
      GLuint getHandle();

   private:
      Array* m_vertices ; 
      Array* m_indices ; 
      GLuint m_handle ; 


};

#endif


