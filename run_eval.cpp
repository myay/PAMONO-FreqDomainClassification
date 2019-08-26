#include <DT_FFT.h>
#include <DT_FWT.h>
#include <DTdatasetFFT.h>
#include <DTdatasetFWT.h>
#include <fftw3.h>

#include <iostream>
#include <chrono>
#include <cmath>
#include <stdlib.h>
#include <dataset.h>
#include <indices.h>
#include <Fwt.h>
#include <vector>
#include <iomanip>

// 1 for debug output enabled, 0 for measuring time
#define DEBUG 0
// set 1 for FFT, 0 for FWT experiment
#define FFT_FWT 1
// size of one row in the input image, here we assume x=y
#define ROWLENGTH 32
// total experiment data length, e.g., 32*32 = 1024 (also for FWT)
#define NFFTW ROWLENGTH*ROWLENGTH

// execution time test paramenters
// how many times the test for execution time should be done
#define NLOOP 1000
// test batch size, when BATCHS number of executions are done, a new input batch will be created
#define BATCHS 25

using namespace std;
using namespace std::chrono;

// arrays for swapping
float swap1a[ROWLENGTH/2 + 1];
float swap2a[ROWLENGTH/2 + 1];

// funtion to draw straight lines
void intline(int x1, int y1, int x2, int y2, float* sangElement, float* outi);

int main(int argc, char *argv[])
{
  if (FFT_FWT)
  {
    std::cout << "Running FFT features experiment:" << std::endl << std::endl;
  }
  else
  {
    std::cout << "Running FWT features experiment:" << std::endl << std::endl;
  }

  Fwt fwt;
  // output of the fft
  fftwf_complex* out = (fftwf_complex*) fftwf_malloc((sizeof(fftwf_complex) * NFFTW));
  fftwf_complex* out_buf = (fftwf_complex*) fftwf_malloc((sizeof(fftwf_complex) * NFFTW));

  // buffers for argument of the fft
  float* out_arg = (float*) fftwf_malloc(sizeof(float) * NFFTW);
  float* out_arg_buf = (float*) fftwf_malloc(sizeof(float) * NFFTW);

  // batch counter, variable to keep track of available test images for batch renewal
  int runs_in_batch = 0;
  // allocate memory for BATCHS * NFFTW random values
  float* in = (float*) fftwf_malloc(sizeof(float) * NFFTW * BATCHS);
  // pointer to move within memory to which "in" points
  float* in_mod = in;

  // FFTW_EXHAUSTIVE will look for the best FFTW implementation for this use case
  fftwf_plan p2;
  p2 = fftwf_plan_dft_r2c_2d(ROWLENGTH, ROWLENGTH, in, out, FFTW_EXHAUSTIVE);

  // set timer for total time measured
  // FFT timers
  std::chrono::nanoseconds total_fft_time(0);
  std::chrono::nanoseconds fft2d_time(0);
  std::chrono::nanoseconds argshift_time(0);
  std::chrono::nanoseconds summations_fft_time(0);
  std::chrono::nanoseconds DT_fft_time(0);
  // FWT timers
  std::chrono::nanoseconds total_wavelet_time(0);
  std::chrono::nanoseconds fwt2d_time(0);
  std::chrono::nanoseconds energy_time(0);
  std::chrono::nanoseconds summations_fwt_time(0);
  std::chrono::nanoseconds DT_wavelet_time(0);

  // classification loop starts here
  for (int runs = 0; runs < NLOOP; runs++)
  {
    srand(runs);
    if (runs == 0 || (runs % BATCHS == 0))
    {
      // create stream of BATCHS images
      // fill experiment batch with random integers from 0 to 255
      for (int it_rand = 0; it_rand < BATCHS*NFFTW; it_rand++)
      {
        // generate random number between 0 and 255
        in[it_rand] = rand() % 256;
      }
      // reset batch counter
      runs_in_batch = 0;
      // reset pointer to beginning of batch array
      in_mod = in;
    }

    #if DEBUG
    {
      std::cout << "Input image:" << std::endl;
      for (int x = 0; x < ROWLENGTH; x++)
      {
        for (int y = 0; y < ROWLENGTH; y++)
        {
          std::cout << in[x*ROWLENGTH + y] << ",";
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
    #endif

    // *******************
    #if  FFT_FWT
    //********************
    // execute 2d fft

    auto start_fft2d = std::chrono::high_resolution_clock::now();

    fftwf_execute(p2);

    fft2d_time += (std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_fft2d));

    auto start_argshift = std::chrono::high_resolution_clock::now();

    // move pointer to next image in batch for next iteration
    in_mod += NFFTW;

    // copy image to standard 2d form
    int iter = 0;
    for (int i = 0; i < ROWLENGTH; i++)
    {
      for (int j = 0; j < ROWLENGTH; j++)
      {
        if (j < ROWLENGTH/2 + 1)
        {
          // copy value
          out_buf[(ROWLENGTH * i) + j][0] = out[iter][0];
          out_buf[(ROWLENGTH * i) + j][1] = out[iter][1];
          iter++;
        }
        else
        {
          // copy a zero
          out_buf[(ROWLENGTH * i) + j][0] = 0;
          out_buf[(ROWLENGTH * i) + j][1] = 0;
        }
      }
    }

    // take sqrt
    for (int i = 0; i < ROWLENGTH; i++)
    {
      for (int j = 0; j < ROWLENGTH/2 + 1; j++)
      {
        if (i < ROWLENGTH/2)
        {
          // square roots
          out_arg[(ROWLENGTH * i) + j] = sqrt((out_buf[ROWLENGTH * i + j][0])*(out_buf[ROWLENGTH * i +j][0]) + (out_buf[ROWLENGTH * i + j][1])*(out_buf[ROWLENGTH * i + j][1]));
        }
        else
        {
          // square roots
          out_arg[(ROWLENGTH * i) + j] = sqrt((out_buf[ROWLENGTH * i + j][0])*(out_buf[ROWLENGTH * i +j][0]) + (out_buf[ROWLENGTH * i + j][1])*(out_buf[ROWLENGTH * i + j][1]));
        }
      }
    }

    // mirror wrt to j
    for (int i = 0; i < ROWLENGTH; i++)
    {
      for (int j = 0; j < ROWLENGTH; j++)
      {
        int index = (ROWLENGTH * i) + j;
        int index_mirrored = (ROWLENGTH * i) + (ROWLENGTH/2 - j);
        if (j < ROWLENGTH/2 + 1)
        {
          out_arg_buf[index_mirrored] = out_arg[index];
        }
        else
        {
          out_arg_buf[index] = 0;
       }
     }
    }

    // swap upper and lower
    float swap_upper;
    float swap_lower;
    for (int i = 0; i < ROWLENGTH/2; i++)
    {
      for (int j = 0; j < ROWLENGTH/2 + 1; j++)
      {
        int index = (ROWLENGTH * i) + j;
        int index_swapped =  (ROWLENGTH * i) + j + ROWLENGTH*ROWLENGTH/2;
        swap_upper = out_arg_buf[index];
        swap_lower = out_arg_buf[index_swapped];
        out_arg[index_swapped] = swap_upper;
        out_arg[index] = swap_lower;
      }
    }

    // swap first row and row ROWLENGTH/2
    for (int i = 1; i < ROWLENGTH/2; i++)
    {
      for (int j = 0; j < ROWLENGTH/2 + 1; j++)
      {
        int index = ROWLENGTH*i + j;
        int index_shifted = ROWLENGTH*(ROWLENGTH - i) + j;
        out_arg_buf[index_shifted] = out_arg[index];
        out_arg_buf[index] = out_arg[index_shifted];
      }
    }

    // change first rows of upper and lower part
    for (int i = 0; i < ROWLENGTH/2 + 1; i++)
    {
      swap1a[i] = out_arg_buf[i];
      swap2a[i] = out_arg_buf[(ROWLENGTH*ROWLENGTH/2) + i];
    }
    for (int i = 0; i < ROWLENGTH/2 + 1; i++)
    {
      out_arg_buf[i] = swap2a[i];
      out_arg_buf[(ROWLENGTH*ROWLENGTH/2) + i] = swap1a[i];
    }

    #if DEBUG
    {
      std::cout << "FFT abs shifted:"<< std::endl;
      for (int x = 0; x < ROWLENGTH; x++)
      {
        for (int y = 0; y < ROWLENGTH; y++)
        {
          float outputv1 = out_arg_buf[x*ROWLENGTH + y];
          std::cout << outputv1 << ", ";
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
    #endif

    argshift_time += (std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_argshift));

    auto start_summations_fft = std::chrono::high_resolution_clock::now();

    // sum on cirles and lines
    // image width
    int width = ROWLENGTH;
    // maximal radius for srad and sang calculation
    int rmax = ROWLENGTH/2 - 1;

    // array for storing srad values
    float srad[rmax];
    float sang[181];
    sang[0] = 0;

    // stores the indices for accessing circle pixels
    int xc[180];
    int yc[180];

    // we do not use this array element
    srad[0] = 0;
    // store DC component here
    srad[1] = out_arg_buf[width*16 + 16];
    // printf("\nout::%0.2f\n", out[width*(specxture->y0) + specxture->x0] );

    // previous indices
    int prev_xc = 0;
    int prev_yc = 0;

    // index in cartesian coordinates
    int index_cart;
    int curr_radius;

    // compute srad. This loop has N/2 iterations (for N*N image)
    for(curr_radius = 2; curr_radius <= rmax; curr_radius++)
    {
      srad[curr_radius] = 0.0f;
      // create halfcircle. Loop in halfcircle has 180 iterations
      // there is a lookup table for the indices which
      // halfcircle(curr_radius, xc, yc, specxture->x0, specxture->y0, out); creates
      // sum on circle. 180 iterations
      for(int i = 0; i < 180; i++)
      {
        // only sum when there is a new coordinate
        //if(i == 0 || prev_xc != xc[i] || prev_yc != yc[i])
        {
          //printf("\nI: %d\n", i);
          //printf("(%d, %d)\n", xc[i], yc[i]);
          index_cart = (sradyc[curr_radius][i])*width + (sradxc[curr_radius][i]);
          //printf("\nout[%d] = %.0f\n", index_cart, out[index_cart]);
          srad[curr_radius] += out_arg_buf[index_cart];
          //prev_xc = xc[i];
          //prev_yc = yc[i];
        }
      }
      //printf("\nsrad[%d] = %0.2f\n", curr_radius, srad[curr_radius]);
    }

    prev_xc = 0;
    prev_yc = 0;

    // calculate sang. xc and yc contain the cartesian coordinates for the circular arc with radius rmax.
    // 180 iterations
    for(int i = 1; i < 180; i++)
    {
      sang[i] = 0;
      // only sum when there is a new coordinate
      //if(i == 0 || prev_xc != xc[i] || prev_yc != yc[i])
      {
        //printf("\nx1,y1:  %d, %d\n", specxture->x0, specxture->y0);
        /******
        int x2 = 2;
        int y2 = 3;
        */
        //printf("\nx2,y2: %d, %d\n", xc[i], yc[i]);
        /*
        if (sradyc[width/2 - 1][i] > 31)
        {
          cout << "ERROR sradc too large in i:" << i << endl;
        }
        */
        intline(width/2, width/2, sradxc[width/2 - 1][i], sradyc[width/2 - 1][i], &sang[i], out_arg_buf);
        //printf("\nI: %d\n", i);
        //printf("\nxc: %d, yc: %d\n", xc[i], yc[i]);
        //printf("\nsang[%d] = %0.2f\n", i, sang[i]);
        //prev_xc = xc[i];
        //prev_yc = yc[i];
      }
    }

    // Spectral features srad
    // max
    float sradMax = srad[1];
    float sradMaxloc = 1;
    for (int i = 2; i < rmax; i++)
    {
      if (srad[i] > sradMax)
      {
        sradMax = srad[i];
        sradMaxloc = i;
      }
    }

    // mean
    float sradMean=0.0f, sradAcc=0.0f;
    for (int i = 1; i < rmax; i++)
    {
      sradAcc += srad[i];
    }
    sradMean = sradAcc / rmax;

    // variance
    float sradVariance=0.0f;
    sradAcc = 0.0f;
    for (int i = 1; i < rmax; i++)
    {
      sradAcc += (srad[i] - sradMean)*(srad[i] - sradMean);
    }
    sradVariance = sradAcc / rmax;

    // distance
    float sradDistance = (sradMean - sradMax) > 0 ? (sradMean - sradMax) : (-1)*(sradMean - sradMax);

    /***************************/
    // spectral features sang
    // max
    float angMax = sang[1];
    float angMaxloc = 1;
    for (int i = 2; i < 181; i++)
    {
      if (sang[i] > angMax)
      {
        angMax = sang[i];
        angMaxloc = i;
      }
    }

    // mean
    float angMean=0.0f, angAcc=0.0f;
    for (int i = 1; i < 181; i++)
    {
      angAcc += sang[i];
    }
    angMean = angAcc / 180;

    // variance
    float angVariance=0.0f;
    angAcc = 0.0f;
    for (int i = 1; i < 181; i++)
    {
      angAcc += (sang[i] - angMean)*(sang[i] - angMean);
    }
    angVariance = angAcc / 180;

    // distance
    float angDistance = (angMean - angMax) > 0 ? (angMean - angMax) : (-1)*(angMean - angMax);

    // spectral feature vector
    float spectralFeatures[10] = {sradMax, sradMaxloc, sradMean, sradVariance, sradDistance, angMax, angMaxloc, angMean, angVariance, angDistance};

    #if DEBUG
    {
      std::cout << "Features:" << std::endl;
      for (int x = 0; x < 10; x++)
      {
        std::cout << spectralFeatures[x] << ",";
      }
      std::cout << std::endl;
      return 0;
    }
    #endif

    summations_fft_time += (std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_summations_fft));

    auto start_DT_fft = std::chrono::high_resolution_clock::now();
    // different feature vectors in every run
    // take features that the DT was trained with
    int DTll = DT_FFT(testDT1000FFT[runs*10 + 0], testDT1000FFT[runs*10 + 1], testDT1000FFT[runs*10 + 2], testDT1000FFT[runs*10 + 3], testDT1000FFT[runs*10 + 4],
    testDT1000FFT[runs*10 + 5], testDT1000FFT[runs*10 + 6], testDT1000FFT[runs*10 + 7], testDT1000FFT[runs*10 + 8], testDT1000FFT[runs*10 + 9]);

    DT_fft_time += (std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_DT_fft));

    // **************************************
    #else
    // **************************************
    // execute wavelet here
    //std::vector<float> in_vector {in, in + 1024};

    // calculate the FWT

    auto start_haarwt = std::chrono::high_resolution_clock::now();

    fwt.haarWT(in, ROWLENGTH, ROWLENGTH, 3);

    #if DEBUG
    {
      std::cout << "FWT:" << std::endl;
      for (int x = 0; x < ROWLENGTH; x++)
      {
        for (int y = 0; y < ROWLENGTH; y++)
        {
          std::cout << in[x*ROWLENGTH + y] << ",";
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
    #endif

    fwt2d_time += (std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_haarwt));

    auto start_energy = std::chrono::high_resolution_clock::now();
    // calculate wavelet energy terms
    for (int i = 0; i < NFFTW; i++)
    {
      in[i] = in[i]*in[i];
    }
    energy_time += (std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_energy));

    auto start_summations_fwt = std::chrono::high_resolution_clock::now();
    // initialize accumulators for energy terms
    float H1=0.0f, D1=0.0f, V1=0.0f, H2=0.0f, D2=0.0f, V2=0.0f, H3=0.0f, D3=0.0f, V3=0.0f, Ea=0.0f;
    // indices for iteration in wavelet decimated regions
    int indexH1=0, indexD1=0, indexV1=0, indexH2=0, indexD2=0, indexV2=0, indexH3=0, indexD3=0, indexV3=0, indexEa=0;

    int width = ROWLENGTH;
    int height = ROWLENGTH;

    int cWidth = ROWLENGTH/2;
    int cHeight = ROWLENGTH/2;

    float Ea1 = 0.0f, Ea2 = 0.0f, Ea3 = 0.0f;

    for (int i = 0; i < cHeight; i++)
    {
      for (int j = 0; j < cWidth; j++)
      {
        indexH1 = width*height/2 + width*i + j;
        H1 += in[indexH1];

        indexD1 = width*(height+1)/2 + width*i + j;
        D1 += in[indexD1];

        indexV1 = width/2 + width*i + j;
        V1 += in[indexV1];

        Ea1 += in[width*i + j];
      }
    }

    float sumAll = H1 + D1 + V1 + Ea1;
    //printf("\nSum all Energies lvl1: %f\n", sumAll);
    //printf("\nEa1: %f\n", Ea1);

    // sum level 2
    cWidth /= 2;
    cHeight /= 2;

    for (int i = 0; i < cHeight; i++)
    {
      for (int j = 0; j < cWidth; j++)
      {
        indexH2 = width*height/4 + width*i + j;
        H2 += in[indexH2];

        indexD2 = width*(height+1)/4 + width*i + j;
        D2 += in[indexD2];

        indexV2 = width/4 + width*i + j;
        V2 += in[indexV2];

        //Ea2 += out[width*i + j];
      }
    }

    //float sumAll2 = H2 + D2 + V2 + Ea2;
    //printf("\nSum all Energies lvl2: %f\n", sumAll);
    //printf("\nEa2: %f\n", Ea2);

    // sum level 3
    cWidth /= 2;
    cHeight /= 2;

    for (int i = 0; i < cHeight; i++)
    {
      for (int j = 0; j < cWidth; j++)
      {
        indexH3 = width*height/8 + width*i + j;
        H3 += in[indexH3];

        indexD3 = width*(height+1)/8 + width*i + j;
        D3 += in[indexD3];

        indexV3 = width/8 + width*i + j;
        V3 += in[indexV3];

        Ea3 += in[width*i + j];
      }
    }

    //float sumAll3 = H3 + D3 + V3 + Ea3;
    //printf("\nSum all Energies lvl3: %f\n", sumAll);
    //printf("\nEa3: %f\n", Ea3);

    float features[10];
    features[0] = (Ea3*100)/sumAll;
    features[1] = (H1*100)/sumAll;
    features[2] = (H2*100)/sumAll;
    features[3] = (H3*100)/sumAll;
    features[4] = (V1*100)/sumAll;
    features[5] = (V2*100)/sumAll;
    features[6] = (V3*100)/sumAll;
    features[7] = (D1*100)/sumAll;
    features[8] = (D2*100)/sumAll;
    features[9] = (D3*100)/sumAll;

    #if DEBUG
    {
      std::cout << "Features:" << std::endl;
      for (int x = 0; x < 10; x++)
      {
        std::cout << features[x] << ",";
      }
      std::cout << std::endl;
      return 0;
    }
    #endif

    summations_fwt_time += (std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_summations_fwt));

    auto start_dt_fwt = std::chrono::high_resolution_clock::now();
    // take features that the DT is actually trained with
    int DTll = DT_FWT(testDT1000FWT[runs*10 + 0], testDT1000FWT[runs*10 + 1], testDT1000FWT[runs*10 + 2], testDT1000FWT[runs*10 + 3], testDT1000FWT[runs*10 + 4],
    testDT1000FWT[runs*10 + 5], testDT1000FWT[runs*10 + 6], testDT1000FWT[runs*10 + 7], testDT1000FWT[runs*10 + 8], testDT1000FWT[runs*10 + 9]);
    DT_wavelet_time += (std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start_dt_fwt));
    /* *** */
    #endif
    /* *** */
    //total += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start);
    //cout << "Prediction: " << DTll << endl;
  }

  total_fft_time = fft2d_time + argshift_time + summations_fft_time + DT_fft_time;
  std::cout << "avg. Total FFT classification time: " << total_fft_time.count()/NLOOP << endl;
  std::cout << "avg. fft2d time: " << fft2d_time.count()/NLOOP << endl;
  std::cout << "avg. argshift fft time: " << argshift_time.count()/NLOOP << endl;
  std::cout << "avg. summations fft time: " << summations_fft_time.count()/NLOOP << endl;
  std::cout << "avg. DT fft time: " <<   DT_fft_time.count()/NLOOP << endl;

  std::cout << "----------" << endl;

  total_wavelet_time = fwt2d_time + energy_time + summations_fwt_time + DT_wavelet_time;
  std::cout << "avg. Total FWT classification time: " << total_wavelet_time.count()/NLOOP << endl;
  std::cout << "avg. fwt2d time: " << fwt2d_time.count()/NLOOP << endl;
  std::cout << "avg. energy time: " << energy_time.count()/NLOOP << endl;
  std::cout << "avg. summations fwt time: " << summations_fwt_time.count()/NLOOP << endl;
  std::cout << "avg. DT fwt time: " <<   DT_wavelet_time.count()/NLOOP << endl;

  //fftwf_free(in);
  //fftwf_free(out);
  //fftwf_free(experiment_batch25);
  return 0;
}

void intline(int x1, int y1, int x2, int y2, float* sangElement, float* outi)
{
  int n = x1*2;
  int flip = 0;
  float m = 0.0;

  int dx = abs(x2 - x1);
  int dy = abs(y2 - y1);
  // no line to draw for this case
  if((dx == 0) && (dy == 0))
  {
    x2 = x1;
    y2 = y1;
    return;
  }

  /*
  for(int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
        out[i*n + j] = 0.0;
    }
  }
  */
  // always take the longer variable, otherwise approximation of the line would be worse
  if(dx >= dy)
  {
    //printf("\nfirst\n");
    if(x1 > x2)
    {
      // swap x1 with x2, and y1 with y2 to always draw from (x1,y1) to (x2,y2), left to right
      int temp;
      temp = x1; x1 = x2; x2 = temp;
      temp = y1; y1 = y2; y2 = temp;
      flip = 1;
    }
    //printf("\nx1,y1: %d, %d | x2,y2: %d, %d\n", x1, y1, x2, y2);
    // calculate gradient
    m = (float)(y2 - y1) / (x2 - x1);
    //printf("\nm = %.2f\n", m);
    // create an array X of length dx which has all values between x1 and x2
    // TODO: replace malloc with 2*N static array, could be faster
    int *x_coord = (int*)malloc((dx+1) * sizeof(int));
    int *y_coord = (int*)malloc((dx+1) * sizeof(int));
    //printf("\n Output of sang:\n");

    //printf("\nx_coord:");
    //printf("\ny_coord:");
    //*sangElement = 0.0f;
    //printf("\nTODO: x1: %d y1: %d \n", x1, y1);
    //printf("\nTODO: x2: %d y2: %d \n", x2, y2);

    for(int i = 0; i <= dx; i++)
    {
      //printf("%d,%d | ", x_coord[i], y_coord[i]);
      x_coord[i] = x1 + i;
      y_coord[i] = round(y1 + m*(x_coord[i]-x1));
      //printf("\nx: %d, y: %d\n", x_coord[i], y_coord[i]);
      //out[y_coord[i]*n + x_coord[i]] = 1.0;
      *sangElement += outi[y_coord[i]*n + x_coord[i]];
    }
    //printf("\n");
    //printf("\n...END...\n");

    /*
    for(int i = 0; i < n; i++)
    {
       for (int j = 0; j < n; j++)
       {
           printf("%.0f, ", out[i*n + j]);
       }
       printf("\n");
    }
    */
    free(x_coord);
    free(y_coord);
    // create another array Y of length dx which calculates y = round(y1 + m*(x - x1)) for all values in X
  }
  else
  {
    //printf("\nsecond\n");
    if(y1 > y2)
    {
      // swap x1 with x2, and y1 with y2 to always draw from (x1,y1) to (x2,y2)
      int temp;
      temp = x1; x1 = x2; x2 = temp;
      temp = y1; y1 = y2; y2 = temp;
      flip = 1;
    }
    //printf("\nx1,y1: %d, %d | x2,y2: %d, %d\n", x1, y1, x2, y2);
    // calculate slope
    m = (float)(x2 - x1) / (y2 - y1);
    // x = round(x1 + m*(y - y1));
    int *y_coord = (int *)malloc((dy+1) * sizeof(int));
    int *x_coord = (int *)malloc((dy+1) * sizeof(int));

    //  printf("\nx_coord:");
    //printf("\n Output of sang:\n");
    //*sangElement = 0.0f;
    //printf("\nTODO: x1: %d y1: %d \n", x1, y1);
    //printf("\nTODO: x2: %d y2: %d \n", x2, y2);
    for(int i = 0; i <= dy; i++)
    {
      y_coord[i] = y1 + i;
      x_coord[i] = round(x1 + m*(y_coord[i]-y1));
      //out[y_coord[i]*n + x_coord[i]] = 1.0;
      //printf("\nx: %d, y: %d\n", x_coord[i], y_coord[i]);
      *sangElement += outi[y_coord[i]*n + x_coord[i]];
    }
    //printf("\n...END...\n");
    /*
    for(int i = 0; i < n; i++)
    {
      for (int j = 0; j < n; j++)
      {
          printf("%.0f, ", out[i*n + j]);
      }
      printf("\n");
    }
    */
    free(x_coord);
    free(y_coord);
  }
}
