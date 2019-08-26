binary=run_eval
all:
	g++ -g -std=c++11 DT_FWT.cpp DT_FFT.cpp run_eval.cpp Fwt.cpp -Iinclude -Llib -o $(binary) -lfftw3f -lm -O3
clean:
	rm -f $(binaries) *.o
