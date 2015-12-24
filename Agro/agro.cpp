#include"opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include"opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include"opencv2/features2d/features2d.hpp"
#include<string.h>
#include<stdio.h>
#include<math.h>
#include<cvblob.h>//for blob detection

/////

#include <vector>
#include <deque>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <direct.h>//_mkdir()

#include "GaborFR.h"
using namespace cvb;


#include <iostream>
	struct pixel_porp{
		int max_intensity;
		int pos;
		//void *cluter_pointer;
	};

const char* keys =
{
	"{i|input| |The source image}"
	"{o|outdir| |The output directory}"
		"{f|filename| |The file name}"
};
struct rect{
	int minx;
	int miny;
	int maxx;
	int maxy;
};
using namespace cv;
using namespace std;

Mat roi(Mat);
int find_max_intensity(Mat cluster1, struct pixel_porp *cluster1_max_pix,int );
Mat morph_operation(Mat );
Mat ROI_IMG;
rect blob_detect(Mat);
Mat gabor_wevlet(Mat);
Mat gw();
//for blob detection
int feature_extraction(IplImage * gray);



#include "GaborFR.h"
void GetLocalEntroyImage(IplImage* gray_src,IplImage* entopy_image);
GaborFR::GaborFR()
{
	isInited = false;
}
void GaborFR::Init(Size ksize, double sigma,double gamma, int ktype)
{
	gaborRealKernels.clear();
	gaborImagKernels.clear();
	double mu[8]={0,1,2,3,4,5,6,7};
	double nu[5]={0,1,2,3,4};
	int i,j;
	for(i=0;i<5;i++)
	{
		for(j=0;j<8;j++)
		{
			gaborRealKernels.push_back(getRealGaborKernel(ksize,sigma,mu[j]*CV_PI/8,nu[i],gamma,ktype));
			gaborImagKernels.push_back(getImagGaborKernel(ksize,sigma,mu[j]*CV_PI/8,nu[i],gamma,ktype));
		}
	}
	isInited = true;
}
Mat GaborFR::getImagGaborKernel(Size ksize, double sigma, double theta, double nu,double gamma, int ktype)
{
	double	sigma_x		= sigma;
	double	sigma_y		= sigma/gamma;
	int		nstds		= 3;
	double	kmax		= CV_PI/2;
	double	f			= cv::sqrt(2.0);
	int xmin, xmax, ymin, ymax;
	double c = cos(theta), s = sin(theta);
	if( ksize.width > 0 )
	{
		xmax = ksize.width/2;
	}
	else//
	{
		xmax = cvRound(std::max(fabs(nstds*sigma_x*c), fabs(nstds*sigma_y*s)));
	}
	if( ksize.height > 0 )
	{
		ymax = ksize.height/2;
	}
	else
	{
		ymax = cvRound(std::max(fabs(nstds*sigma_x*s), fabs(nstds*sigma_y*c)));
	}
	xmin = -xmax;
	ymin = -ymax;
	CV_Assert( ktype == CV_32F || ktype == CV_64F );
	float*	pFloat;
	double*	pDouble;
	Mat kernel(ymax - ymin + 1, xmax - xmin + 1, ktype);
	double k		=	kmax/pow(f,nu);
	double scaleReal=	k*k/sigma_x/sigma_y;
	for( int y = ymin; y <= ymax; y++ )
	{
		if( ktype == CV_32F )
		{
			pFloat = kernel.ptr<float>(ymax-y);
		}
		else
		{
			pDouble = kernel.ptr<double>(ymax-y);
		}
		for( int x = xmin; x <= xmax; x++ )
		{
			double xr = x*c + y*s;
			double v = scaleReal*exp(-(x*x+y*y)*scaleReal/2);
			double temp=sin(k*xr);
			v	=  temp*v;
			if( ktype == CV_32F )
			{
				pFloat[xmax - x]= (float)v;
			}
			else
			{
				pDouble[xmax - x] = v;
			}
		}
	}
	return kernel;
}
//sigma
Mat GaborFR::getRealGaborKernel( Size ksize, double sigma, double theta,
	double nu,double gamma, int ktype)
{
	double	sigma_x		= sigma;
	double	sigma_y		= sigma/gamma;
	int		nstds		= 3;
	double	kmax		= CV_PI/2;
	double	f			= cv::sqrt(2.0);
	int xmin, xmax, ymin, ymax;
	double c = cos(theta), s = sin(theta);
	if( ksize.width > 0 )
	{
		xmax = ksize.width/2;
	}
	else//
	{
		xmax = cvRound(std::max(fabs(nstds*sigma_x*c), fabs(nstds*sigma_y*s)));
	}

	if( ksize.height > 0 )
		ymax = ksize.height/2;
	else
		ymax = cvRound(std::max(fabs(nstds*sigma_x*s), fabs(nstds*sigma_y*c)));
	xmin = -xmax;
	ymin = -ymax;
	CV_Assert( ktype == CV_32F || ktype == CV_64F );
	float*	pFloat;
	double*	pDouble;
	Mat kernel(ymax - ymin + 1, xmax - xmin + 1, ktype);
	double k		=	kmax/pow(f,nu);
	double exy		=	sigma_x*sigma_y/2;
	double scaleReal=	k*k/sigma_x/sigma_y;
	int	   x,y;
	for( y = ymin; y <= ymax; y++ )
	{
		if( ktype == CV_32F )
		{
			pFloat = kernel.ptr<float>(ymax-y);
		}
		else
		{
			pDouble = kernel.ptr<double>(ymax-y);
		}
		for( x = xmin; x <= xmax; x++ )
		{
			double xr = x*c + y*s;
			double v = scaleReal*exp(-(x*x+y*y)*scaleReal/2);
			double temp=cos(k*xr) - exp(-exy);
			v	=	temp*v;
			if( ktype == CV_32F )
			{
				pFloat[xmax - x]= (float)v;
			}
			else
			{
				pDouble[xmax - x] = v;
			}
		}
	}
	return kernel;
}
Mat GaborFR::getMagnitude(Mat &real,Mat &imag)
{
	CV_Assert(real.type()==imag.type());
	CV_Assert(real.size()==imag.size());
	int ktype=real.type();
	int row = real.rows,col = real.cols;
	int i,j;
	float*	pFloat,*pFloatR,*pFloatI;
	double*	pDouble,*pDoubleR,*pDoubleI;
	Mat		kernel(row, col, real.type());
	for(i=0;i<row;i++)
	{
		if( ktype == CV_32FC1 )
		{
			pFloat = kernel.ptr<float>(i);
			pFloatR= real.ptr<float>(i);
			pFloatI= imag.ptr<float>(i);
		}
		else
		{
			pDouble = kernel.ptr<double>(i);
			pDoubleR= real.ptr<double>(i);
			pDoubleI= imag.ptr<double>(i);
		}
		for(j=0;j<col;j++)
		{
			if( ktype == CV_32FC1 )
			{
				pFloat[j]= sqrt(pFloatI[j]*pFloatI[j]+pFloatR[j]*pFloatR[j]);
			}
			else
			{
				pDouble[j] = sqrt(pDoubleI[j]*pDoubleI[j]+pDoubleR[j]*pDoubleR[j]);
			}
		}
	}
	return kernel;
}
Mat GaborFR::getPhase(Mat &real,Mat &imag)
{
	CV_Assert(real.type()==imag.type());
	CV_Assert(real.size()==imag.size());
	int ktype=real.type();
	int row = real.rows,col = real.cols;
	int i,j;
	float*	pFloat,*pFloatR,*pFloatI;
	double*	pDouble,*pDoubleR,*pDoubleI;
	Mat		kernel(row, col, real.type());
	for(i=0;i<row;i++)
	{
		if( ktype == CV_32FC1 )
		{
			pFloat = kernel.ptr<float>(i);
			pFloatR= real.ptr<float>(i);
			pFloatI= imag.ptr<float>(i);
		}
		else
		{
			pDouble = kernel.ptr<double>(i);
			pDoubleR= real.ptr<double>(i);
			pDoubleI= imag.ptr<double>(i);
		}
		for(j=0;j<col;j++)
		{
			if( ktype == CV_32FC1 )
			{
// 				if(pFloatI[j]/(pFloatR[j]+pFloatI[j]) > 0.99)
// 				{
// 					pFloat[j]=CV_PI/2;
// 				}
// 				else
// 				{
//					pFloat[j] = atan(pFloatI[j]/pFloatR[j]);
				pFloat[j] = asin(pFloatI[j]/sqrt(pFloatR[j]*pFloatR[j]+pFloatI[j]*pFloatI[j]));
/*				}*/
//				pFloat[j] = atan2(pFloatI[j],pFloatR[j]);
			}//CV_32F
			else
			{
				if(pDoubleI[j]/(pDoubleR[j]+pDoubleI[j]) > 0.99)
				{
					pDouble[j]=CV_PI/2;
				}
				else
				{
					pDouble[j] = atan(pDoubleI[j]/pDoubleR[j]);
				}
//				pDouble[j]=atan2(pDoubleI[j],pDoubleR[j]);
			}//CV_64F
		}
	}
	return kernel;
}
Mat GaborFR::getFilterRealPart(Mat& src,Mat& real)
{
	CV_Assert(real.type()==src.type());
	Mat dst;
	Mat kernel;
	flip(real,kernel,-1);//
//	filter2D(src,dst,CV_32F,kernel,Point(-1,-1),0,BORDER_CONSTANT);
	filter2D(src,dst,CV_32F,kernel,Point(-1,-1),0,BORDER_REPLICATE);
	return dst;
}
Mat GaborFR::getFilterImagPart(Mat& src,Mat& imag)
{
	CV_Assert(imag.type()==src.type());
	Mat dst;
	Mat kernel;
	flip(imag,kernel,-1);//
//	filter2D(src,dst,CV_32F,kernel,Point(-1,-1),0,BORDER_CONSTANT);
	filter2D(src,dst,CV_32F,kernel,Point(-1,-1),0,BORDER_REPLICATE);
	return dst;
}
void GaborFR::getFilterRealImagPart(Mat& src,Mat& real,Mat& imag,Mat &outReal,Mat &outImag)
{
	outReal=getFilterRealPart(src,real);
	outImag=getFilterImagPart(src,imag);
}

int gabor(Mat I)
{
	//Mat M = getGaborKernel(Size(9,9),2*CV_PI,u*CV_PI/8, 2*CV_PI/pow(2,CV_PI*(v+2)/2),1,0);
	Mat saveM;
	//s8-4
	//s1-5
	//s1
	//Mat I=imread("H:\\pic\\s1-5.bmp",-1);
	normalize(I,I,1,0,CV_MINMAX,CV_32F);
	Mat showM,showMM;Mat M,MatTemp1,MatTemp2;
	Mat outR,outI;
	IplImage srcEntropy = outR;
	char c='a';
	Mat line;
	Mat lineOut;
	Mat Mout,MMout;
	Mat M2;
	int iSize=50;
	IplImage *outEntropy;
	for(int i=1;i<=8;i++)
	{
		showM.release();
		for(int j=0;j<5;j++)
		{
			Mat M1= GaborFR::getRealGaborKernel(Size(iSize,iSize),2*CV_PI,i*CV_PI/8+CV_PI/2, j,1);
			Mat M2 = GaborFR::getImagGaborKernel(Size(iSize,iSize),2*CV_PI,i*CV_PI/8+CV_PI/2, j,1);


			GaborFR::getFilterRealImagPart(I,M1,M2,outR,outI);
			//cout<<"hh1";
//			M=GaborFR::getPhase(M1,M2);
//			M=GaborFR::getMagnitude(M1,M2);
//			M=GaborFR::getPhase(outR,outI);
//			M=GaborFR::getMagnitude(outR,outI);
 //			M=GaborFR::getMagnitude(outR,outI);
// 			MatTemp2=GaborFR::getPhase(outR,outI);
// 			M=outR;
			 M=M2;
			// 		resize(M,M,Size(100,100));


			normalize(M,M,0,255,CV_MINMAX,CV_8U);
			showM.push_back(M);
			line=Mat::ones(4,M.cols,M.type())*255;
			showM.push_back(line);


			string s1;
			s1="F:\\IP_Project\\OCWS\\Hello\\out_img\\";
			c++;
			s1+=c;
			s1=s1+".jpg";
			//cout<<s1;
			//imshow(s1,outR);
			//A.convertTo(B,CV_8U);
			Mat Mip;
			normalize(outR,Mip,0,255,CV_MINMAX,CV_8U);
			//outR.convertTo(Mip,CV_8U);
			cout<<endl<<cout<<"Depth : "<< Mip.depth();
		IplImage ip =Mip;


		//cout<<"6st"<<ip.imageData<<endl;

			feature_extraction( &ip);

		//	feature_extraction( &srcEntropy );
			//Mat abc=Mat(outEntropy);
			//imshow(s1, abc);
		}
		showM=showM.t();
		line=Mat::ones(4,showM.cols,showM.type())*255;
		showMM.push_back(showM);
		showMM.push_back(line);

	}
	showMM=showMM.t();


//	bool flag=imwrite("H:\\out.bmp",showMM);
	imshow("saveMM",showMM);
	cout<<"hello";
	waitKey(0);
	return 0;
}
int main(int argc, char ** argv){

	CommandLineParser parser(argc, argv, keys);
	string infile = parser.get<string>("input");
	string outdir = parser.get<string>("outdir");
	string filename = parser.get<string>("filename");

	Mat src = imread(infile,CV_LOAD_IMAGE_UNCHANGED);

	if (src.empty())
		{

			return -1;
		}
	Mat src_lab;
	cvtColor(src, src_lab, CV_BGR2Lab);


//namedWindow( "Input image", 200 );
//imshow( "Input image", src );





	Mat roi_image= roi(src);

	Mat gray_image;

	//int x;



	cvtColor(roi_image, gray_image, CV_BGR2GRAY);
	cout<<endl<<" "<< gray_image.depth();
	IplImage ip = gray_image;

		cout<<endl<<"1st "<<ip.depth;
		cout<<"1st "<<ip.BorderConst;
		cout<<"1st "<<ip.BorderMode;
		cout<<"2st "<<ip.alphaChannel;
		cout<<"3st "<<ip.channelSeq;
		cout<<"4st "<<ip.colorModel;
		cout<<"5st "<<ip.dataOrder;
	//	cout<<"6st"<<ip.imageData<<endl;

	//feature_extraction( &ip );

	/*	gray_image.convertTo(gray_image, CV_8UC1);
	for(int i=1000; i<gray_image.cols+1000; i++) {
					x=gray_image.at<int>(i/gray_image.cols, i%gray_image.cols);
					cout<<x<<" ";
			}
*/		Mat img_bw= gray_image> 50;//threshold by 50 for disease cluster
Scalar intensity;

/*		cv::threshold(gray_image, img_bw, 30, 255, cv::THRESH_BINARY);
cout<<gray_image.type();
cout<<" "<<img_bw.type();

		for(int i=1000; i<gray_image.cols+1000; i++) {
				x=img_bw.at<int>(i/gray_image.cols, i%gray_image.cols);
				cout<<x<<" ";
		}
	*/
//namedWindow( "BinaryOfDisease", 200 );
//imshow( "BinaryOfDisease", img_bw );
	Mat after_morph= Mat(roi_image.rows, roi_image.cols, CV_8UC1);

	after_morph=morph_operation(img_bw);

/*	for(int i=0; i<after_morph.cols*after_morph.rows; i++) {
		x=after_morph.at<int>(i/after_morph.cols, i%after_morph.cols);
		if(x!=0){
		//	after_morph.at<Vec3b>(i/after_morph.cols, i%after_morph.cols)[0]=1.0;
		}
}

	namedWindow( "aM", 200 );
	imshow( "aM", after_morph );



	 const Mat& mask=Mat();

*/

	   Mat cluster1=Mat(roi_image.rows, roi_image.cols, CV_8UC3);

	roi_image.convertTo(roi_image,CV_8UC3);
Vec3b vec;




for(int i=0; i<after_morph.rows; i++) {
		for(int j=0;j<after_morph.cols;j++){
				intensity=after_morph.at<uchar>(i,j);


				if(intensity[0]!=0){

					vec = (ROI_IMG.at<Vec3b>(i,j));

					cluster1.at<Vec3b>(i,j)[0]=vec[0];
					cluster1.at<Vec3b>(i,j)[1]=vec[1];
					cluster1.at<Vec3b>(i,j)[2]=vec[2];

			}
		}
	}
//namedWindow( "DiseasePart", 200 );
//imshow( "DiseasePart", cluster1);

	rect cord=blob_detect(after_morph);
	 Rect cropRect = Rect(cord.minx,cord.miny,cord.maxx-cord.minx,cord.maxy-cord.miny); // ROI in source image
		   		//Rect cropRect = Rect(100, 100, 255, 255);
		   		Mat largest_blob=cluster1(cropRect);
//namedWindow( "largest_blob", 200 );
//imshow( "largest_blob", largest_blob);
//cout<<"out";
Mat gw_image;
Mat lb_gray;
cvtColor(largest_blob,lb_gray,CV_RGB2GRAY);
	gabor(lb_gray);

	imwrite( filename, largest_blob);

//namedWindow( "final", CV_WINDOW_AUTOSIZE );
//imshow( "final", roi_image );


	waitKey(0);
	return 0;

}


Mat roi(Mat img){

	int top_row,top_col,bottom_row,bottom_col;
	int left_col,left_row,right_row,right_col;
	int crop_row;
	int crop_col;



	Mat filtered_image= Mat(img.size(), CV_8UC1);
	Mat org_image = img.clone();




	bilateralFilter(img, filtered_image, 0, 20.0, 2.0);

//namedWindow( "Filtered image", CV_WINDOW_AUTOSIZE );
//imshow( "Filtered image", filtered_image );

	Mat gray_image;
	cvtColor( filtered_image, gray_image, CV_BGR2GRAY );

//namedWindow( "After smooth", CV_WINDOW_AUTOSIZE );
//imshow( "After smooth", gray_image );

Mat img_bw;
threshold( gray_image, img_bw, 140, 255,THRESH_BINARY );
	//Mat img_bw = gray_image> 128;// verify from sir


//namedWindow( "Binary", CV_WINDOW_AUTOSIZE );
//imshow( "Binary", img_bw );

	int row=img_bw.rows;
	int col=img_bw.cols;

	//cout <<"Total row "<<row<<endl;
	//cout <<"Toral col "<<col<<endl;
	int flag=0;
	int i;
	int j;
	Scalar intensity;


	intensity[0]=255;
	for( i=0;i<row;i++){
		for( j=0;j<col;j++){
			intensity=img_bw.at<uchar>(i,j);

			if(intensity[0]==0){
				flag=1;
				break;
			}
		}
		if(flag==1)break;
	}
	if(flag==1){


		top_row=i;
		top_col=j;

		//cout<<"tr"<<top_row<<endl;
		//cout<<"tc"<<top_col<<endl;
	}
	intensity[0]=255;
	flag=0;
	for( i=row-1;i>0;i--){
			for( j=0;j<col;j++){
				intensity=img_bw.at<uchar>(i,j);
					if(intensity[0]==0){
					flag=1;
					break;
				}
			}
			if(flag==1)break;
	}
	if(flag==1){
		bottom_row=i;
		bottom_col=j;
		//cout<<"br"<<bottom_row<<endl;
		//cout<<"bc"<<bottom_col<<endl;
	}
	intensity[0]=255;
	flag=0;
	for( j=0;j<col;j++){
		for( i=0;i<row;i++){
			intensity=img_bw.at<uchar>(i,j);
					if(intensity[0]==0){
					flag=1;
					break;
				}
			}
			if(flag==1)break;
		}
		if(flag==1){
			left_row=i;
			left_col=j;

			//cout<<"lr"<<left_row<<endl;
			//cout<<"lc"<<left_col<<endl;
	}
	intensity[0]=255;
	flag=0;
	for( j=col-1;j>0;j--){
		for( i=0;i<row;i++){
			intensity=img_bw.at<uchar>(i,j);
					if(intensity[0]==0){
					flag=1;
					break;
				}
			}
			if(flag==1)break;
		}
		if(flag==1){
			right_row=i;
			right_col=j;

			//cout<<"rr"<<right_row<<endl;
			//cout<<"rc"<<right_col<<endl;
		}
		crop_row=bottom_row-top_row;
		crop_col=right_col-left_col;



		Rect cropRect = Rect(left_col,top_row,crop_col,crop_row); // ROI in source image
		//Rect cropRect = Rect(100, 100, 255, 255);
		Mat roi_image=img(cropRect);

//namedWindow("roi",CV_WINDOW_AUTOSIZE);
//imshow("roi",roi_image);

	ROI_IMG=img(cropRect);

		Mat bestLabels,centers,clustered;

		Mat p = Mat::zeros(roi_image.cols*roi_image.rows, 5, CV_32F);	//zero(row,col,type)initialize with zeros
		vector<Mat> bgr;
		    cv::split(roi_image, bgr);	//split images in channel bgr[0], bgr[1], bgr[2] no of channel in roi_image=3
		    // i think there is a better way to split pixel bgr color
		    for(int i=0; i<roi_image.cols*roi_image.rows; i++) {
		        p.at<float>(i,0) = (i/roi_image.cols) / roi_image.rows;  // row value / #of row
		        p.at<float>(i,1) = (i%roi_image.cols) / roi_image.cols;		// col value /#of col
		        p.at<float>(i,2) = bgr[0].data[i] / 255.0;
		        p.at<float>(i,3) = bgr[1].data[i] / 255.0;
		        p.at<float>(i,4) = bgr[2].data[i] / 255.0;
		    }
		int K = 3;
		//  printf("test 2\n");
		kmeans(p, K, bestLabels,
		            TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0),
		            3, KMEANS_PP_CENTERS, centers);

		//   printf("test 2\n");
		   clustered = Mat(roi_image.rows, roi_image.cols, CV_32F);
		  // Mat cluster1;
		   //Mat cluster1 = CopyOneImage(roi_image);
		   //roi_image.copyTo(cluster1);
		   Mat cluster1=Mat(roi_image.rows, roi_image.cols, CV_8UC3);
		   Mat cluster2=Mat(roi_image.rows, roi_image.cols, CV_32FC3);
		   Mat cluster3=Mat(roi_image.rows, roi_image.cols, CV_32FC3);



		   int colors[K];
		   		    for(int i=0; i<K; i++) {
		   		        colors[i] = 255/(i+1);
		   		    }
		       for(int i=0; i<roi_image.cols*roi_image.rows; i++) {
		           clustered.at<float>(i/roi_image.cols, i%roi_image.cols) = (float)(colors[bestLabels.at<int>(0,i)]);
		           if((bestLabels.at<int>(0,i))==0){

		        	   cluster1.at<Vec3b>(i/roi_image.cols, i%roi_image.cols) = (roi_image.at<Vec3b>(i/roi_image.cols,i%roi_image.cols));

		        	   cluster2.at<Vec3f>(i/roi_image.cols, i%roi_image.cols)[0] = 0.0;
					   cluster2.at<Vec3f>(i/roi_image.cols, i%roi_image.cols)[1] = 0.0;
					   cluster2.at<Vec3f>(i/roi_image.cols, i%roi_image.cols)[2] = 0.0;

					   cluster3.at<Vec3f>(i/roi_image.cols, i%roi_image.cols)[0] = 0.0;
					   cluster3.at<Vec3f>(i/roi_image.cols, i%roi_image.cols)[1] = 0.0;
					   cluster3.at<Vec3f>(i/roi_image.cols, i%roi_image.cols)[2] = 0.0;


		           }
		           if((bestLabels.at<int>(0,i))==1){
							cluster2.at<Vec3f>(i/roi_image.cols, i%roi_image.cols) = (roi_image.at<Vec3b>(i/roi_image.cols,i%roi_image.cols));

							cluster1.at<Vec3b>(i/roi_image.cols, i%roi_image.cols)[0] = 0.0;
							cluster1.at<Vec3b>(i/roi_image.cols, i%roi_image.cols)[1] = 0.0;
							cluster1.at<Vec3b>(i/roi_image.cols, i%roi_image.cols)[2] = 0.0;

							cluster3.at<Vec3f>(i/roi_image.cols, i%roi_image.cols)[0] = 0.0;
							cluster3.at<Vec3f>(i/roi_image.cols, i%roi_image.cols)[1] = 0.0;
							cluster3.at<Vec3f>(i/roi_image.cols, i%roi_image.cols)[2] = 0.0;
		          		           }
		           if((bestLabels.at<int>(0,i))==2){
		          		        	   cluster3.at<Vec3f>(i/roi_image.cols, i%roi_image.cols) =(roi_image.at<Vec3b>(i/roi_image.cols,i%roi_image.cols));

		          		        	   cluster2.at<Vec3f>(i/roi_image.cols, i%roi_image.cols) [0] = 0.0;
		          		        	   cluster2.at<Vec3f>(i/roi_image.cols, i%roi_image.cols) [1]= 0.0;
		          		        	   cluster2.at<Vec3f>(i/roi_image.cols, i%roi_image.cols) [2]= 0.0;


		          		        	   cluster1.at<Vec3b>(i/roi_image.cols, i%roi_image.cols)[0] = 0.0;
		          		        	   cluster1.at<Vec3b>(i/roi_image.cols, i%roi_image.cols)[1] = 0.0;
		          		        	   cluster1.at<Vec3b>(i/roi_image.cols, i%roi_image.cols)[2] = 0.0;
		          		           }


		       }



		      // cout<<roi_image.rows<<" "<<roi_image.cols;
		      	//	   cout<<clustered.rows<<" "<<clustered.cols;
		      //		   cout<<cluster1.rows<<" "<<cluster1.cols;
		      // for(int i=0; i<roi_image.cols*roi_image.rows; i++) {
		    //	   cluster1.at<int>(i/roi_image.cols, i%roi_image.cols) = roi_image.at<int>(i/roi_image.cols,i%roi_image.cols);
		      // }


/*
clustered.convertTo(clustered, CV_8U);
imshow("clustered", clustered);
*/



//cluster1.convertTo(cluster1, CV_8UC3);

	pixel_porp cluster1_max_pix;
	pixel_porp cluster2_max_pix;
	pixel_porp cluster3_max_pix;


	 find_max_intensity(cluster1, &cluster1_max_pix,0);
	 find_max_intensity(cluster2, &cluster2_max_pix,0);
	 find_max_intensity(cluster3, &cluster3_max_pix,0);

		pixel_porp *first_min_pix;
		pixel_porp *second_min_pix;
 Mat* fm,*sm;

	 if(cluster1_max_pix.max_intensity<cluster2_max_pix.max_intensity){
		 if(cluster2_max_pix.max_intensity<cluster3_max_pix.max_intensity){
			 first_min_pix=&cluster1_max_pix;fm=&cluster1;
			 second_min_pix=&cluster2_max_pix;sm=&cluster2;
		 }
		 else{
			 if(cluster3_max_pix.max_intensity<cluster1_max_pix.max_intensity){
				 first_min_pix=&cluster3_max_pix;fm=&cluster3;
				 second_min_pix=&cluster1_max_pix;sm=&cluster1;
			 }
			 else{
				 first_min_pix=&cluster1_max_pix;fm=&cluster1;
				 second_min_pix=&cluster3_max_pix;sm=&cluster3;
			 }
		 }
	 }
	 else{
		 if(cluster1_max_pix.max_intensity<cluster3_max_pix.max_intensity){
					 first_min_pix=&cluster2_max_pix;fm=&cluster2;
					 second_min_pix=&cluster1_max_pix;sm=&cluster1;
				 }
				 else{
					 if(cluster3_max_pix.max_intensity<cluster2_max_pix.max_intensity){
						 first_min_pix=&cluster3_max_pix;fm=&cluster3;
						 second_min_pix=&cluster2_max_pix;sm=&cluster2;
					 }
					 else{
						 first_min_pix=&cluster2_max_pix;fm=&cluster2;
						 second_min_pix=&cluster3_max_pix;sm=&cluster3;
					 }
				 }
	 }
cout<<"FI-"<<cluster1_max_pix.max_intensity<<endl;
cout<<"SI-"<<cluster2_max_pix.max_intensity<<endl;
cout<<"TI-"<<cluster3_max_pix.max_intensity<<endl;
cout<<"FMin-"<<first_min_pix->max_intensity<<endl;
cout<<"SMin-"<<second_min_pix->max_intensity<<endl;


pixel_porp first_max_pix,second_max_pix;

find_max_intensity(*fm, &first_max_pix,1);
find_max_intensity(*sm, &second_max_pix,1);

int ratio;

if(first_min_pix->max_intensity>second_min_pix->max_intensity)
	ratio=first_min_pix->max_intensity/second_min_pix->max_intensity;
else
	ratio=second_min_pix->max_intensity/first_min_pix->max_intensity;

Mat * outImage;

if(ratio>3.5){
	if((((first_min_pix->pos)>(second_min_pix->pos))||((second_min_pix->pos)>(first_min_pix->pos)))&&((second_max_pix.pos)>(first_max_pix.pos))){
		outImage=sm;
	}
	else outImage=fm;
}
else outImage=fm;


outImage=fm;
	outImage->convertTo(*outImage, CV_8UC3);

	sm->convertTo(*sm, CV_8UC3);


/*
namedWindow( "First min", 200 );
imshow("First min",*fm);

namedWindow( "Second min", 200 );
imshow("Second min",*sm);


namedWindow( "ROI", 200 );
imshow("ROI",*outImage);

namedWindow( "x", CV_WINDOW_AUTOSIZE );
		imshow( "x", roi_image );

*/





				/*	vector<Mat> bgr_planes;
					cv::split(cluster1, bgr_planes);

					int histSize = 256;

					  // Set the ranges
					  float range[] = { 0, 256 } ;
					  const float* histRange = { range };
					  bool uniform = true; bool accumulate = false;

					  Mat g_hist;

					  // Compute the histograms:
					   calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );

					  int temp=0;
					  int count=0;
					  int pos=-1;

					  for( int i = 1; i < histSize; i++ ){
						  temp=g_hist.at<float>(i);
						  if( count<temp) {
							  count=temp;

							  pos=i;
						  }

					  }
					  cout<<cluster3.rows*cluster3.cols<<"** " ;
					  cout<<count<<" "<<pos;

*/













					/*		       		 Mat result=Mat(channels[0]);


		       		double minVal; double maxVal; Point minLoc; Point maxLoc;

		       		Point matchLoc;

		       		    minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

		       		 cout<<"MaxI- "<<maxVal<<" "<<" "<<"MinI- "<< minVal<<endl;

		       	  for(int i=10; i<roi_image.cols*25; i++) {
		       		 // cout<<channels[0].data[i];
		       		  cout<<result.at<Vec3b>(i/roi_image.cols, i%roi_image.cols)<<" ";
		       		  cout <<cluster1.at<Vec3b>(i/roi_image.cols, i%roi_image.cols)<<"#";
		       		  i=i5;
		       			      }
*/




//Mat cls1;
//cvtColor(cluster1, cls1, CV_Lab2RGB);

//namedWindow( "cluster1", CV_WINDOW_AUTOSIZE );
//imshow("cluster1", cluster1);

//cluster2.convertTo(cluster2, CV_8UC3);
//namedWindow( "cluster2", CV_WINDOW_AUTOSIZE );
//imshow("cluster2", cluster2);

//cluster3.convertTo(cluster3, CV_8UC3);
//namedWindow( "cluster3", CV_WINDOW_AUTOSIZE );
//imshow("cluster3", cluster3);

		/*	cvSetImageROI(&img2, cropRect);

		cvCopy((IplImage)img, roi_image, NULL); // Copies only crop region
		cvResetImageROI((IplImage)img);
*/
	/*	roi_image(crop_row,crop_col,CV_8UC3);
		 namedWindow( "R image", CV_WINDOW_AUTOSIZE );
				 imshow( "R image", roi_image );

		for(i=top_row;i<=bottom_row;i++){
			for(j=left_col;i<=left_col;i++){
				Vec3b intensity=img.at<Vec3b>(i,j);
				Vec3b &intensityToBe=roi_image.at<Vec3b>(i-top_row,j-left_col);
				for(int k=0;k<img.channels();k++){
					intensityToBe.val[k]=intensity.val[k];
				}
			}
		}

*/
 return *outImage;

}

int  find_max_intensity(Mat cluster1, struct pixel_porp *cluster1_max_pix,int flag){

						vector<Mat> bgr_planes;
						cv::split(cluster1, bgr_planes);

						int histSize = 256;

						  // Set the ranges
						  float range[] = { 0, 256 } ;
						  const float* histRange = { range };
						  bool uniform = true; bool accumulate = false;

						  Mat g_hist;

						  // Compute the histograms:
						   calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
						  int count=0;
						  int pos=-1;
						  if(flag==1){
							  int temp=0;
							 for(int i=255;i>=50;i--){
								 temp=g_hist.at<float>(i);
								 if(temp!=0){
									 count=temp;
									 pos=i;
									 break;
								 }
							 }

						  }
						  else{
							  int temp=0;

							  for( int i = 1; i < histSize; i++ ){
								  temp=g_hist.at<float>(i);
								  if( count<temp) {
									  count=temp;
									  pos=i;
								  }
							  }

						  }

						 // cout<<cluster3.rows*cluster3.cols<<"** " ;
						  cluster1_max_pix->max_intensity=count;
						  cluster1_max_pix->pos=pos;
						//  cluster1_max_pix->cluter_pointer=cluster1;
						  cout<<count<<" "<<pos;
return 0;
}
Mat morph_operation(Mat src){
	/*
	Mat dila_dst,erod_dst;


	 Mat element = getStructuringElement( MORPH_CROSS,
	                                       Size(3, 3 ),
	                                       Point( -1, -1 ) );
	  dilate( src, dila_dst, element );
namedWindow( "Dilated", 200 );
imshow("Dilated", dila_dst);
	element = getStructuringElement( MORPH_CROSS,
	                                       Size(5, 5 ),
	                                       Point( -1, -1 ) );
	  erode( dila_dst, erod_dst, element );

namedWindow( "Dila_Erod", 200 );
imshow("Dila_Erod", erod_dst);


	  //	erode( src,  dst, InputArray kernel, Point anchor=Point(-1,-1), int iterations=1, int borderType=BORDER_CONSTANT, const Scalar& borderValue=morphologyDefaultBorderValue() )

	 return erod_dst;

	*/
	Mat dst;
	 Mat element = getStructuringElement( MORPH_CROSS,
		                                       Size(11, 11 ),
		                                       Point( -1, -1 ) );
	 const Scalar& borderValue=morphologyDefaultBorderValue();
	morphologyEx(src, dst, MORPH_CLOSE, element, Point(-1,-1),1,BORDER_CONSTANT,borderValue );
//namedWindow( "close", 200 );
//imshow("close", dst);
	Mat dst1;
	  element = getStructuringElement( MORPH_CROSS,
			                                       Size(11, 11 ),
			                                       Point( -1, -1 ) );
	morphologyEx(dst, dst1, MORPH_OPEN, element, Point(-1,-1),1,BORDER_CONSTANT,borderValue );
//namedWindow( "open", 200 );
//imshow("open", dst1);
	return dst1;

}

rect blob_detect(Mat src1){



/*
	cv::SimpleBlobDetector::Params params;
	params.minDistBetweenBlobs = 1.0f;
	params.filterByInertia = false;
	params.filterByConvexity = false;
	params.filterByColor = false;
	params.filterByCircularity = false;
	params.filterByArea = true;
	params.minArea = 1;
	params.maxArea = 320000;
	// ... any other params you don't want default value

	// set up and create the detector using the parameters
	cv::Ptr<cv::FeatureDetector> blob_detector = new cv::SimpleBlobDetector(params);
	blob_detector->create("SimpleBlob");

	// detect!
	vector<cv::KeyPoint> keypoints;
	vector<cv::KeyPoint> temp;
	blob_detector->detect(src, keypoints);

	// extract the x y coordinates of the keypoints:
	int size=0;
	int x,y,s;

	cout<<endl<<endl;
	for (int i=0; i<keypoints.size(); i++){
		 int X=keypoints[i].pt.x;
		 int Y=keypoints[i].pt.y;

		if(size<keypoints[i].size){
			x=X;
			y=Y;
			size=keypoints[i].size;
		}

			cout<<X<<" "<<Y<<" ";
			cout<<keypoints[i].size<<endl;

			for(int i=Y;i<Y+5;i++){
			          					for(int j=X;j<X+5;j++){
			          						src.at<uchar>(i,j)=0;

			          					}
			          				}


	}
	cout<<"largest blob"<<endl;
		cout<<x<<" "<<y<<"size- "<<size<<endl;
          cout<<src.rows<<" "<<src.cols<<endl;
          for(int i=y;i<y+20;i++){
          					for(int j=x;j<x+20;j++){
          						src.at<uchar>(i,j)=0;

          					}
          				}

		namedWindow( "blb", 200 );
		imshow("blb", src);

	return src;
	*/

	IplImage src=src1;

	 IplImage *labelImg = cvCreateImage(cvGetSize(&src),IPL_DEPTH_LABEL,1);

	 CvBlobs blobs;

	   unsigned int result = cvLabel(&src, labelImg, blobs);
	//   cvRenderBlobs(labelImg, blobs, &src, &src);
	   IplImage *imgOut = cvCreateImage(cvGetSize(&src), IPL_DEPTH_8U, 3); cvZero(imgOut);
	  CvLabel label=cvLargestBlob(blobs);
	  if(label!=0) {
		  // Delete all blobs except the largest
		 cvFilterByLabel(blobs, label);

		}
//cout<<endl<<label;
int minx,miny,maxx,maxy;
rect cord;
for (CvBlobs::const_iterator it=blobs.begin(); it!=blobs.end(); ++it)
	   {

	   //  cout << it->second->minx<<" "<<it->second->miny<<it->second->maxx<<" "<<it->second->maxy<< endl;
	     cord.minx=it->second->minx;
	     cord.miny=it->second->miny;
	     cord.maxx=it->second->maxx;
	     cord.maxy=it->second->maxy;
	   }




	 //  cvb::cvFilterLabels (imgIn, IplImage *imgOut, const CvBlobs &blobs)
	//  IplImage *imgOut2;
	//   cvRenderBlobs(labelImg, blobs, &src, imgOut);
	 // cvb::cvFilterLabels (imgOut,imgOut2,blobs);


//  cvNamedWindow("blobed",200);
//  imshow("blobed", largest_blob);
	  // cvShowImage("blobed",largest_blob);



/*	   //unsigned int i = 0;

	   // Render contours:
	   cout<<"hello2";
	   for (CvBlobs::const_iterator it=blobs.begin(); it!=blobs.end(); ++it)
	   {
	     //cvRenderBlob(labelImg, (*it).second, img, imgOut);

	     CvScalar meanColor = cvBlobMeanColor((*it).second, labelImg, &src);
	   //  cout << "Mean color: r=" << (unsigned int)meanColor.val[0] << ", g=" << (unsigned int)meanColor.val[1] << ", b=" << (unsigned int)meanColor.val[2] << endl;

	     CvContourPolygon *polygon = cvConvertChainCodesToPolygon(&(*it).second->contour);

	     CvContourPolygon *sPolygon = cvSimplifyPolygon(polygon, 10.);
	     CvContourPolygon *cPolygon = cvPolygonContourConvexHull(sPolygon);
cout<<"hello3";
	     cvRenderContourChainCode(&(*it).second->contour, imgOut);
	     cvRenderContourPolygon(sPolygon, imgOut, CV_RGB(0, 0, 255));
	     cvRenderContourPolygon(cPolygon, imgOut, CV_RGB(0, 255, 0));
cout<<"Heloo4";
delete cPolygon;
	     delete sPolygon;
	     delete polygon;
cout<<"heloo3";
	     // Render internal contours:
	     for (CvContoursChainCode::const_iterator jt=(*it).second->internalContours.begin(); jt!=(*it).second->internalContours.end(); ++jt)
	       cvRenderContourChainCode((*jt), imgOut);

	     //stringstream filename;
	     //filename << "blob_" << setw(2) << setfill('0') << i++ << ".png";
	     //cvSaveImageBlob(filename.str().c_str(), imgOut, (*it).second);
	   }

	   cvNamedWindow("test", 1);
	   cvShowImage("test", imgOut);
	   //cvShowImage("grey", grey);
	   cvWaitKey(0);
	   cvDestroyWindow("test");

	  // cvReleaseImage(&imgOut);
	  // cvReleaseImage(&grey);
	  // cvReleaseImage(&labelImg);
	 //  cvReleaseImage(&img);

	   cvReleaseBlobs(blobs);
	   */

	   return cord;
}

Mat gabor_wevlet(Mat in1){
	Mat in;
	cvtColor(in1, in, CV_BGR2GRAY);
	 //cvtColor(in1,in,CV_RGB2GRAY); // load grayscale
	Mat dest;
	Mat src_f;
	in.convertTo(src_f,CV_32F);

	int kernel_size = 64;
	double sig = 1, th = 0, lm = 1.0, gm = 0.02, ps = 0;
	cout<<"h1";
	//for()
	cv::Mat kernel = cv::getGaborKernel(cv::Size(kernel_size,kernel_size), sig, th, lm, gm, ps);
	cout<<"h2";
	cv::filter2D(src_f, dest, CV_32F, kernel);
	cout<<"h3";
	cerr << dest(Rect(30,30,10,10)) << endl; // peek into the data

	Mat viz;
	dest.convertTo(viz,CV_8U,1.0/255.0);     // move to proper[0..255] range to show it
	imshow("k",kernel);
	imshow("d",viz);
	waitKey();

	return in1;



}

void GetLocalEntroyImage(IplImage* gray_src,IplImage* entopy_image){

int hist_size[]={256};
float gray_range[]={0,255};
float* ranges[] = { gray_range};
    CvHistogram * hist = cvCreateHist( 1, hist_size, CV_HIST_SPARSE, ranges,1);
    for(int i=0;i<gray_src->width;i++){
            for(int j=0;j<gray_src->height;j++){
                //calculate entropy for pixel(i,j)
                //1.set roi rect(9*9),handle edge pixel
                CvRect roi;
                int threshold=max(0,i-4);
                roi.x=threshold;
                threshold=max(0,j-4);
                roi.y=threshold;
                roi.width=(i-max(0,i-4))+1+(min(gray_src->width-1,i+4)-i);
                roi.height=(j-max(0,j-4))+1+(min(gray_src->height-1,j+4)-j);
                cvSetImageROI(const_cast<IplImage*>(gray_src),roi);
                IplImage *gray_src_non_const=const_cast<IplImage*>(gray_src);

                //2.calHist,here I chose CV_HIST_SPARSE to speed up
                cvCalcHist( &gray_src_non_const, hist, 0, 0 );
                cvNormalizeHist(hist,1.0);
                float total=0;
                float entroy=0;

               //3.get entroy

               // entopy_image = cvCreateImage( cvGetSize(src), IPL_DEPTH_64F, 1 );
                //Mat entopy_image(A, B, CV_32FC1);

                CvSparseMatIterator it;
                for(CvSparseNode*node=cvInitSparseMatIterator((CvSparseMat*)hist->bins,&it);node!=0;node=cvGetNextSparseNode(&it))
                {
                  float gray_frequency=*(float*)CV_NODE_VAL((CvSparseMat*)hist->bins,node);
                  entroy=entroy-gray_frequency*(log(gray_frequency)/log(2.0f));//*(log(gray_frequency)/log(2.0))
                }
               ((float *)((entopy_image)->imageData + (entopy_image)->widthStep*j))[i]=entroy;

               //(((float *)((entopy_image)->imageData + (entopy_image)->widthStep*(j)))[i]) = entroy;

                //   entopy_image.at<float>(i, j) = entroy;
                //cvReleaseHist(&hist);

            }
        }
        cvResetImageROI(const_cast<IplImage*>(gray_src));
    }

