#ifndef FWT_H
#define FWT_H

#include <vector>

class Fwt {
  public:
    void haarWT(float* features, unsigned int patchWidth, unsigned int patchHeight, unsigned int levels);
  private:
    void dwt1level_vec(float* input, int width);
    void dwt1level_vec_rows(float* input, int offset, int width);
};

#endif
