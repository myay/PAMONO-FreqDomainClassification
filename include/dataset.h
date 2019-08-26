/*
 * Some test arrays for FFT classifiction on ARM cortex a 53
 */

#ifndef DATASET_H
#define DATASET_H

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

float in1024[] = { 97,44,117,214,93,142,93,46,44,243,38,61,138,193,124,64,175,4,213,238,6,126,23,199,58,206,208,83,130,249,188,33,
195,30,62,97,68,185,154,255,237,215,130,213,116,32,147,236,35,131,244,19,120,77,95,103,252,36,44,67,150,198,139,189,
230,33,239,153,101,92,127,122,26,216,86,197,188,11,5,58,232,37,144,198,201,160,46,206,8,175,241,108,108,176,159,239,
9,110,221,9,211,93,65,90,206,234,119,149,180,208,103,49,37,70,101,239,5,196,221,248,127,171,208,101,5,168,190,110,
154,253,149,141,8,157,163,221,140,167,72,80,11,178,240,68,78,114,182,206,213,237,119,151,62,178,79,25,243,251,228,157,
96,215,26,224,120,14,55,119,228,90,155,159,133,166,6,29,119,203,56,241,169,190,114,12,5,139,115,207,208,206,2,213,
103,124,102,208,108,66,179,90,5,179,163,62,88,167,60,234,225,241,174,50,143,182,239,205,46,217,142,113,232,85,246,235,
114,254,225,149,254,134,196,237,209,165,221,23,139,0,117,226,186,23,207,112,96,142,169,110,221,108,232,84,4,177,125,218,
104,103,49,243,45,103,92,249,208,151,99,170,238,61,185,207,250,206,132,148,230,14,121,47,101,111,20,223,214,148,152,98,
80,155,154,49,49,129,170,26,98,15,135,33,82,130,182,238,132,83,110,197,27,232,147,253,64,238,133,201,120,89,233,8,
199,238,115,49,87,134,237,114,60,106,107,40,97,66,70,141,166,184,81,229,13,223,182,200,92,125,227,254,0,248,80,225,
215,170,133,249,166,238,233,155,192,20,219,141,77,203,60,253,116,112,70,195,94,223,177,247,155,183,132,25,40,77,47,3,
16,124,127,183,225,244,169,61,82,30,124,50,252,38,126,227,227,158,163,161,83,254,236,238,252,129,25,233,8,120,208,95,
49,253,86,112,212,160,13,239,240,8,232,34,29,129,143,225,168,162,141,120,105,39,214,3,122,150,98,47,108,34,149,105,
192,109,212,62,230,50,111,155,227,30,56,137,57,3,48,101,124,75,178,5,8,17,22,35,67,192,137,206,88,110,219,154,
54,123,240,177,57,202,196,131,254,95,254,248,77,22,147,41,253,252,192,13,134,199,193,154,43,190,193,129,129,15,41,239,
161,63,77,41,25,96,44,9,150,226,76,75,65,185,241,182,130,28,228,0,126,251,118,172,17,175,183,111,232,157,90,210,
47,35,242,100,56,79,216,45,17,76,119,154,228,30,255,2,248,50,233,193,187,147,232,113,142,78,142,38,205,114,200,176,
20,21,254,86,54,8,26,180,178,197,67,112,98,61,250,31,171,238,42,252,209,8,58,114,72,48,40,174,224,204,241,160,
29,154,199,53,210,91,195,224,29,62,131,143,74,230,235,188,232,84,202,147,214,183,32,170,186,108,200,229,141,96,23,71,
146,38,214,1,26,222,179,67,195,34,248,235,170,185,38,199,246,168,116,68,2,185,16,49,215,252,90,44,90,50,121,39,
96,77,15,139,50,230,90,151,157,161,33,102,214,240,206,248,175,162,197,245,178,38,138,96,86,18,154,7,43,124,111,72,
142,135,111,154,204,2,136,190,196,2,91,253,117,172,185,143,83,40,2,170,47,121,219,88,226,8,40,235,194,94,32,82,
98,150,171,22,119,89,45,201,114,111,114,233,66,125,120,141,134,225,238,199,187,204,169,6,132,195,111,22,55,139,206,200,
30,184,37,244,106,81,34,11,132,29,171,39,45,71,142,181,15,190,17,51,23,19,63,99,205,144,132,131,205,196,63,190,
23,140,75,218,64,4,4,81,89,144,193,169,233,13,91,208,43,148,178,110,194,196,18,85,136,247,204,176,12,190,248,163,
149,19,160,196,28,58,155,93,32,90,245,202,10,6,25,36,21,231,246,140,176,73,98,136,165,145,94,227,51,8,178,132,
215,130,64,111,177,171,16,212,167,210,63,231,82,134,224,143,233,187,50,181,214,8,5,249,32,145,48,182,10,9,214,33,
208,68,43,110,196,151,221,134,223,202,50,240,138,69,181,64,32,241,155,85,175,93,151,166,166,142,220,253,42,217,101,35,
177,2,240,48,137,154,50,114,37,201,156,141,28,18,70,179,241,105,241,78,29,125,211,43,29,24,206,100,167,164,237,124,
169,44,31,68,129,249,26,63,188,31,230,164,91,226,245,20,75,117,19,12,158,88,44,43,125,55,57,246,194,177,27,169,
27,240,249,69,65,23,142,249,112,167,196,156,149,75,182,120,97,19,169,52,235,162,206,155,222,24,249,7,115,101,45,82 };

float in16[] = {97.3,44,117,214,93,142,93,46,44,243,38,61,138,193,124,64};

#ifdef __cplusplus
}  /* extern "C" */
#endif /* __cplusplus */

#endif /* DATASET_H */