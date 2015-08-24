
//  https://github.com/glfw/glfw/issues/4

#include <GLFW/glfw3.h>
#include <stdio.h>

int main()
{
  glfwInit();

  GLFWwindow *window = glfwCreateWindow(1440, 900, "", glfwGetPrimaryMonitor(), NULL);
  glfwMakeContextCurrent(window);

  int window_width, window_height, framebuffer_width, framebuffer_height;
  glfwGetWindowSize(window, &window_width, &window_height);
  glfwGetFramebufferSize(window, &framebuffer_width, &framebuffer_height);
  printf("window: %d,%d\n", window_width, window_height);
  printf("framebuffer: %d,%d\n", framebuffer_width, framebuffer_height);

  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
