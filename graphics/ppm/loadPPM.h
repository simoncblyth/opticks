#ifndef LOADPPM_H
#define LOADPPM_H

#ifdef __cplusplus
extern "C" {
#endif

extern unsigned char* loadPPM(char* path, int* width, int* height);

#ifdef __cplusplus
}
#endif


#ifdef LOADPPM_IMPLEMENTATION

// http://www.udel.edu/CIS/220/jatlas/09FH/project1/imagerender/TextureLoader.h

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

unsigned char* loadPPM(char* path, int* width, int* height) {
  FILE* fp;
  int i, w, h, d;
  unsigned char* image;
  char head[70]; /* max line <= 70 in PPM (per spec). */

  fp = fopen(path, "rb");
  if (!fp) {
    //perror(path);
    return NULL;
  }

  /* grab first two chars of the file and make sure that it has the
   correct magic cookie for a raw PPM file. */
  fgets(head, 70, fp);
  if (strncmp(head, "P6", 2)) {
    fprintf(stderr, "%s: Not a raw PPM file\n", path);
    return NULL;
  }

  /* grab the three elements in the header (width, height, maxval). */
  i = 0;
  while (i < 3) {
    fgets(head, 70, fp);
    if (head[0] == '#') /* skip comments. */
      continue;
    if (i == 0)
      i += sscanf(head, "%d %d %d", &w, &h, &d);
    else if (i == 1)
      i += sscanf(head, "%d %d", &h, &d);
    else if (i == 2)
      i += sscanf(head, "%d", &d);
  }

  /* grab all the image data in one fell swoop. */
  image = (unsigned char*)malloc(sizeof(unsigned char)*w*h*3);
  fread(image, sizeof(unsigned char), w*h*3, fp);
  fclose(fp);

  *width = w;
  *height = h;
  return image;
}

#endif
#endif



