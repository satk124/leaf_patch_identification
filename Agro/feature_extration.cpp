
#include <iostream>
#include <fstream>
#include "cv.h"
#include "highgui.h"
#include "cl_Texture.h"
using namespace std;

std::string IntToString(int i)
{
char* buffer = new char[100];
sprintf(buffer, "%d", i);
string s  = string(buffer);
delete [] buffer;
return s;
}
int feature_extraction(IplImage * gray )
{
	cvNamedWindow( "Original Image", 1 );
	cvShowImage( "Original Image",gray );


///IplImage* pImg;
//IplImage* gray;
//string imagePath;
//cvNamedWindow( "ROI", 1 );
//cvNamedWindow( "Original Image", 1 );
int count_steps=1;
const int StepDirections[] = { 0,1,-1,1,-1,0,-1,-1 };
cl_Texture* texture=new cl_Texture( );
cl_Texture::GLCM* glcm;
double d0, d1, d2, d3;
double * features = new double[4*count_steps]; // store the features in this array
int x,y; // to change the ROI
x=y= 0;
//ofstream myfile ("data.txt",ios::out | ios::app );

//if (!myfile.is_open()) cout << "Unable to open file";
//bool closeWindow = true;
int key;
int nummer = 1;

// loading the image
//imagePath= "C:\\Users\\sat\\Pictures\\diseased_dataset\\2\\a.jpg";
//pImg = cvLoadImage(imagePath.c_str(), 1);
if(gray == 0)
{
cout<<"Error!!"<<endl;
return -1;
}
// convert it to gray image
//gray=cvCreateImage(cvSize(pImg->width,pImg->height),pImg->depth,1);
//cvCvtColor(pImg,gray,CV_RGB2GRAY);

//cvNamedWindow( "Original Image", 1 );
//cvShowImage( "Original Image",gray );
cout<<"h1";

// set the ROI in the image
//cvSetImageROI(gray,cvRect(86+x,75+y,75,62));
glcm=texture->CreateGLCM(gray, 1,StepDirections,count_steps,CV_GLCM_OPTIMIZATION_LUT); // buid the GLCM
cout<<"h2";
texture->CreateGLCMDescriptors(glcm, CV_GLCMDESC_OPTIMIZATION_ALLOWDOUBLENEST); // get the features from GLCM
cout<<"h3";
/***************************************************************
***************** Use the features of GLCM  ********************
****************************************************************
*/
for(int i =0; i<count_steps;i++)
{
d0=texture->GetGLCMDescriptor(glcm,i,CV_GLCMDESC_ENTROPY);
features[i*count_steps] = d0;
d1=texture->GetGLCMDescriptor(glcm,i,CV_GLCMDESC_ENERGY);
features[i*count_steps+1] = d1;
d2=texture->GetGLCMDescriptor(glcm,i,CV_GLCMDESC_HOMOGENITY);
features[i*count_steps+2] = d2;
d3=texture->GetGLCMDescriptor(glcm,i,CV_GLCMDESC_CONTRAST);
features[i*count_steps+3] = d3;
cout<< "Entropy: "<<d0<<" Energy: "<<d1<<" Homogenity: "<<d2<<" Contrast: "<<d3<<endl;
//myfile<<d0<<' '<<d1<<' '<<d2<<' '<<d3<<' ';
}
//cout<<endl;
//myfile<< "\n";



//myfile.close();
//cvDestroyWindow( "ROI" );
//cvDestroyWindow( "Original Image" );
//cvReleaseImage( &pImg );
//cvReleaseImage(&gray);
//delete[] features;

return 0;
}
