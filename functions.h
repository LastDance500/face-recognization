#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#include <opencv2/core.hpp>
#include <string>
#include <vector>


using namespace std;
using namespace cv;


string getTime();

int cameraWindow();

void facecollect();

void scan_one_dir( const char * dir_name, int dirnum, const char * dir_txt);

int getcsv();

//static Mat norm_0_255(InputArray _src);

void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';');

void facetrain();

void facerec();


#endif // FUNCTIONS_H
