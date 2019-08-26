#include <Fwt.h>
#include <cmath>
#include <cstring>
#include <stdlib.h>

using namespace std;
// use std::vector<float> my_vector {arr, arr + arr_length};
void Fwt::haarWT(float* features, unsigned int patchWidth, unsigned int patchHeight, unsigned int levels)
{
  int w = patchWidth;
  int h = patchHeight;
  for (unsigned int level = 0; level < levels; level++)
  {
    // rows
    for (int row = 0; row < h; row++)
    {
      int offset = patchWidth * row;
      dwt1level_vec_rows(features, offset, w);
    }

    // columns
    float* temp = (float*) malloc(sizeof(float) * h);
    for (int col = 0; col < w; col++)
    {
      for (int row = 0; row < h; row++)
      {
        temp[row] = features[col + patchWidth * row];
      }
      dwt1level_vec(temp, w);
      for (int row = 0; row < h; row++)
      {
        features[col + patchWidth * row] = temp[row];
      }
    }
    w /= 2;
    h /= 2;
    free(temp);
  }
}

void Fwt::dwt1level_vec(float* input, int width)
{
  std::vector<float> temp(width);
  int n = width / 2;

  for (int i = 0; i < n; i++)
  {
    temp[i] = input[i*2] + input[i*2 + 1];
    temp[i] /= 2;

    temp[i + n] = input[i*2] - input[i*2 + 1];
    temp[i + n] /= 2;
  }
  memcpy(&input[0], &temp[0], width * sizeof(float));
}

void Fwt::dwt1level_vec_rows(float* input, int offset, int width)
{
  std::vector<float> temp(width);
  int n = width / 2;

  for (int i = 0; i < n; i++)
  {
    int i2 = offset + 2 * i; // access correct row in features array with offset

    temp[i] = input[i2] + input[i2 + 1];
    temp[i] /= 2;

    temp[i + n] = input[i2] - input[i2 + 1];
    temp[i + n] /= 2;
  }
  memcpy(&input[offset], &temp[0], width * sizeof(float));
}
