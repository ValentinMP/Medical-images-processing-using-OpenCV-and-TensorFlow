#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <math.h>
#include <dcmtk/dcmimgle/dcmimage.h>
#include <sstream> //for image parsing
#include <time.h> // for execution time measurment

#ifndef SYSOUT_F
#define SYSOUT_F(f, ...)      _RPT1( 0, f, __VA_ARGS__ ) // For Visual studio
#endif

#ifndef speedtest__             
#define speedtest__(data)   for (long blockTime = NULL; (blockTime == NULL ? (blockTime = clock()) != NULL : false); SYSOUT_F(data "%.9fs", (double) (clock() - blockTime) / CLOCKS_PER_SEC))
#endif
// you need these includes for the function
#include <windows.h> // for windows systems

#include <algorithm>    // std::sort

using namespace cv;
using namespace std;

void GammaCorrection(Mat& src, Mat& dst, float fGamma)

{

	unsigned char lut[256];

	for (int i = 0; i < 256; i++)

	{

		lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);

	}

	dst = src.clone();

	const int channels = dst.channels();

	switch (channels)

	{

	case 1:

	{

		MatIterator_<uchar> it, end;

		for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)

			*it = lut[(*it)];

		break;

	}

	case 3:

	{

		MatIterator_<Vec3b> it, end;

		for (it = dst.begin<Vec3b>(), end = dst.end<Vec3b>(); it != end; it++)

		{

			(*it)[0] = lut[((*it)[0])];

			(*it)[1] = lut[((*it)[1])];

			(*it)[2] = lut[((*it)[2])];

		}

		break;

	}

	}

}

/**
 *  \brief Automatic brightness and contrast optimization with optional histogram clipping
 *  \param [in]src Input image GRAY or BGR or BGRA
 *  \param [out]dst Destination image
 *  \param clipHistPercent cut wings of histogram at given percent tipical=>1, 0=>Disabled
 *  \note In case of BGRA image, we won't touch the transparency
*/
void BrightnessAndContrastAuto(Mat& src, Mat& dst, float clipHistPercent)
{

	CV_Assert(clipHistPercent >= 0);
	CV_Assert((src.type() == CV_8UC1) || (src.type() == CV_8UC3) || (src.type() == CV_8UC4));

	int histSize = 256;
	float alpha, beta;
	double minGray = 0, maxGray = 0;

	//to calculate grayscale histogram
	cv::Mat gray;
	if (src.type() == CV_8UC1) gray = src;
	else if (src.type() == CV_8UC3) cvtColor(src, gray, cv::COLOR_RGB2GRAY);
	else if (src.type() == CV_8UC4) cvtColor(src, gray, cv::COLOR_RGB2GRAY);
	if (clipHistPercent == 0)
	{
		// keep full available range
		cv::minMaxLoc(gray, &minGray, &maxGray);
	}
	else
	{
		cv::Mat hist; //the grayscale histogram

		float range[] = { 0, 256 };
		const float* histRange = { range };
		bool uniform = true;
		bool accumulate = false;
		calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

		// calculate cumulative distribution from the histogram
		std::vector<float> accumulator(histSize);
		accumulator[0] = hist.at<float>(0);
		for (int i = 1; i < histSize; i++)
		{
			accumulator[i] = accumulator[i - 1] + hist.at<float>(i);
		}

		// locate points that cuts at required value
		float max = accumulator.back();
		clipHistPercent *= (max / 100.0); //make percent as absolute
		clipHistPercent /= 2.0; // left and right wings
		// locate left cut
		minGray = 0;
		while (accumulator[minGray] < clipHistPercent)
			minGray++;

		// locate right cut
		maxGray = histSize - 1;
		while (accumulator[maxGray] >= (max - clipHistPercent))
			maxGray--;
	}

	// current range
	float inputRange = maxGray - minGray;

	alpha = (histSize - 1) / inputRange;   // alpha expands current range to histsize range
	beta = -minGray * alpha;             // beta shifts current range so that minGray will go to 0

	// Apply brightness and contrast normalization
	// convertTo operates with saurate_cast
	src.convertTo(dst, -1, alpha, beta);

	// restore alpha channel from source 
	if (dst.type() == CV_8UC4)
	{
		int from_to[] = { 3, 3 };
		cv::mixChannels(&src, 4, &dst, 1, from_to, 1);
	}
	return;
}

void insertionSort(int window[])
{
	int temp, i, j;
	for (i = 0; i < 9; i++) {
		temp = window[i];
		for (j = i - 1; j >= 0 && temp < window[j]; j--) {
			window[j + 1] = window[j];
		}
		window[j + 1] = temp;
	}
}

void medianFilter(Mat& source, Mat& destination)
{

	if (source.type() == CV_8UC1) { destination = source.clone(); cout << "CV_8UC1"; }
	else if (source.type() == CV_8UC3) {
		cvtColor(source, destination, cv::COLOR_RGB2GRAY); cout << "CV_8UC3";
	}
	else if (source.type() == CV_8UC4) {
		cvtColor(source, destination, cv::COLOR_RGB2GRAY); cout << "CV_8UC4";
	}


	int window[9];

	destination = source.clone();
	for (int y = 0; y < source.rows; y++)
		for (int x = 0; x < source.cols; x++)
			for (int c = 0; c < source.channels(); c++)
				destination.at<Vec3b>(y, x)[c] = 0.0;

	for (int y = 1; y < source.rows - 1; y++) {
		for (int x = 1; x < source.cols - 1; x++) {


			// Pick up window element

			window[0] = source.at<Vec3b>(y - 1, x - 1)[1];
			window[1] = source.at<Vec3b>(y, x - 1)[1];
			window[2] = source.at<Vec3b>(y + 1, x - 1)[1];
			window[3] = source.at<Vec3b>(y - 1, x)[1];
			window[4] = source.at<Vec3b>(y, x)[1];
			window[5] = source.at<Vec3b>(y + 1, x)[1];
			window[6] = source.at<Vec3b>(y - 1, x + 1)[1];
			window[7] = source.at<Vec3b>(y, x + 1)[1];
			window[8] = source.at<Vec3b>(y + 1, x + 1)[1];

			// sort the window to find median
			insertionSort(window);

			// assign the median to centered element of the matrix
			destination.at<Vec3b>(y, x)[1] = window[4];

		}
	}

	for (int y = 1; y < source.rows - 1; y++) {
		for (int x = 1; x < source.cols - 1; x++) {


			// Pick up window element

			window[0] = source.at<Vec3b>(y - 1, x - 1)[2];
			window[1] = source.at<Vec3b>(y, x - 1)[2];
			window[2] = source.at<Vec3b>(y + 1, x - 1)[2];
			window[3] = source.at<Vec3b>(y - 1, x)[2];
			window[4] = source.at<Vec3b>(y, x)[2];
			window[5] = source.at<Vec3b>(y + 1, x)[2];
			window[6] = source.at<Vec3b>(y - 1, x + 1)[2];
			window[7] = source.at<Vec3b>(y, x + 1)[2];
			window[8] = source.at<Vec3b>(y + 1, x + 1)[2];

			// sort the window to find median
			insertionSort(window);

			// assign the median to centered element of the matrix
			destination.at<Vec3b>(y, x)[2] = window[4];

		}
	}

	for (int y = 1; y < source.rows - 1; y++) {
		for (int x = 1; x < source.cols - 1; x++) {


			// Pick up window element

			window[0] = source.at<Vec3b>(y - 1, x - 1)[3];
			window[1] = source.at<Vec3b>(y, x - 1)[3];
			window[2] = source.at<Vec3b>(y + 1, x - 1)[3];
			window[3] = source.at<Vec3b>(y - 1, x)[3];
			window[4] = source.at<Vec3b>(y, x)[3];
			window[5] = source.at<Vec3b>(y + 1, x)[3];
			window[6] = source.at<Vec3b>(y - 1, x + 1)[3];
			window[7] = source.at<Vec3b>(y, x + 1)[3];
			window[8] = source.at<Vec3b>(y + 1, x + 1)[3];

			// sort the window to find median
			insertionSort(window);

			// assign the median to centered element of the matrix
			destination.at<Vec3b>(y, x)[1] = window[4];

		}
	}

	return;
}

// Gaussian filtering with code

float distance(int x, int y, int i, int j) {
	return float(sqrt(pow(x - i, 2) + pow(y - j, 2)));
}

double gaussian(float x, double sigma) {
	return exp(-(pow(x, 2)) / (2 * pow(sigma, 2))) / (2 * CV_PI * pow(sigma, 2));

}

void applyBilateralFilter(Mat source, Mat filteredImage, int x, int y, int diameter, double sigmaI, double sigmaS) {
	double iFiltered = 0;
	double wP = 0;
	int neighbor_x = 0;
	int neighbor_y = 0;
	int half = diameter / 2;

	for (int i = 0; i < diameter; i++) {
		for (int j = 0; j < diameter; j++) {
			neighbor_x = x - (half - i);
			neighbor_y = y - (half - j);
			double gi = gaussian(source.at<uchar>(neighbor_x, neighbor_y) - source.at<uchar>(x, y), sigmaI);
			double gs = gaussian(distance(x, y, neighbor_x, neighbor_y), sigmaS);
			double w = gi * gs;
			iFiltered = iFiltered + source.at<uchar>(neighbor_x, neighbor_y) * w;
			wP = wP + w;
		}
	}
	iFiltered = iFiltered / wP;
	filteredImage.at<double>(x, y) = iFiltered;


}

Mat bilateralFilterOwn(Mat source, int diameter, double sigmaI, double sigmaS) {
	Mat filteredImage = Mat::zeros(source.rows, source.cols, CV_64F);
	int width = source.cols;
	int height = source.rows;

	for (int i = 2; i < height - 2; i++) {
		for (int j = 2; j < width - 2; j++) {
			applyBilateralFilter(source, filteredImage, i, j, diameter, sigmaI, sigmaS);
		}
	}
	return filteredImage;
}
float var(int hist[],int level,float val,int pix_num )
{
    long long total=pix_num*val;
    int n=0;
    long long m=0;
    for(int i=0;i<level;i++)
    {
        m+=i*hist[i];
        n+=hist[i];
    }
    long long rem=total-m;
    int rempix=pix_num-n;
    float w0=(1.0*n)/(1.0*pix_num);
    float w1=(1.0*rem)/(1.0*pix_num);
    float u0=(1.0*m)/(1.0*n);
    float u1=(1.0*rem)/(1.0*rempix);
    return w0*w1*(u0-u1)*(u0-u1);
}
void imgSeg(Mat &input_image, Mat &output_image)
{

    long long u=0;
    int hist[256];
    for(int i=0;i<256;i++)
        hist[i]=0;
    int sz=input_image.cols*input_image.rows;
    for (int i=0;i<input_image.rows;i++)
    {
        for(int j=0;j<input_image.cols;j++)
        {
            int n=input_image.at<uchar>(i,j);
            u+=n;
            hist[n]++;
        }
    }
    int pix_num=input_image.rows*input_image.cols;
    float val=(1.0*u)/float(pix_num);
    float max=0;
    int threshold=0;
    for(int i=1;i<255;i++)
    {
        int x=var(hist,i,val,pix_num);
        if(x>max)
        {
            max=x;
            threshold=i;
        }
    }
    for(int i=0;i<input_image.rows;i++)
    {
        for(int j=0;j<input_image.cols;j++)
        {
            if(input_image.at<uchar>(i,j)>threshold)
            {
                input_image.at<uchar>(i,j)=255;
            }
            else
                input_image.at<uchar>(i,j)=0;
        }
    }

    output_image = input_image.clone();
 
}

//image parsing from input folder

/* Returns a list of files in a directory (except the ones that begin with a dot) */
int readFilenames(std::vector<string> &filenames, const string &directory)
{

    HANDLE dir;
    WIN32_FIND_DATA file_data;

    if ((dir = FindFirstFile((directory + "\\*.png").c_str(), &file_data)) == INVALID_HANDLE_VALUE)
        return -1; /* No files found */

    do {
        const string file_name = file_data.cFileName;
        const string full_file_name = directory + "\\" + file_name;
        const bool is_directory = (file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;

        if (file_name[0] == '.')
            continue;

        if (is_directory)
            continue;

        filenames.push_back(full_file_name);
    } while (FindNextFile(dir, &file_data));

    FindClose(dir);

    std::sort (filenames.begin(), filenames.end()); //optional, sort the filenames
    return(filenames.size()); //Return how many we found
} // GetFilesInDirectory

int main()
{
	auto start = chrono::steady_clock::now();
	Mat new_image_gamma_corr, new_image_auto, image_segm, new_image_without_noise;
	/*String imageName("images\\input\\example2.png");
	
	

//single image processing
	image = imread(samples::findFile(imageName), IMREAD_GRAYSCALE);
	if (image.empty())                      // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	GammaCorrection(image, new_image_gamma_corr, 0.5);
	BrightnessAndContrastAuto(new_image_gamma_corr, new_image_auto, 4);
	//imwrite("images\\output\\example1_AutoBrightnessAndContrast.png", new_image_auto);
	//BrightnessAndContrastAuto(new_image_gamma_corr, image_just_with_bright, 4);
	
	//medianFilter(new_image_auto, new_image_without_noise);
	bilateralFilter(new_image_auto, new_image_without_noise, 15, 80, 80);
	//new_image_without_noise = bilateralFilterOwn(new_image_auto, 15, 80, 80);
	//imwrite("images\\output\\example1_bilateralFilterOwn.png", new_image_without_noise);
	//imwrite("images\\output\\example1Prepo.png", new_image_auto);
     imgSeg(new_image_without_noise, image_segm);
	
	imwrite("images\\output\\example2.png", image_segm);

	
	waitKey(0);
	*/
//multiple images processing form a folder


    string folder = "images\\input";
    cout << "Reading in directory " << folder << endl;
    vector<string> filenames;

    int num_files = readFilenames(filenames, folder);
    cout << "Number of files = " << num_files << endl;
	cout << "Folder: " << folder;
    for(int i = 0; i < num_files; ++i)
    {
		string file =filenames[i];
		cout << file;
       Mat image = imread(samples::findFile(file), IMREAD_GRAYSCALE);
		cout << image.size() << endl;

        if(image.empty()) { //Protect against no file
            cerr << folder + filenames[i] << ", file #" << i << ", is not an image" << endl;
            continue;
        }
		
		GammaCorrection(image, new_image_gamma_corr, 0.5);
		BrightnessAndContrastAuto(new_image_gamma_corr, new_image_auto, 0);
		//imwrite("images\\output\\example1_AutoBrightnessAndContrast.png", new_image_auto);
		//BrightnessAndContrastAuto(new_image_gamma_corr, image_just_with_bright, 4);

		//medianFilter(new_image_auto, new_image_without_noise);
		bilateralFilter(new_image_auto, new_image_without_noise, 15, 80, 80);
		//new_image_without_noise = bilateralFilterOwn(new_image_auto, 15, 80, 80);
		//imwrite("images\\output\\example1_bilateralFilterOwn.png", new_image_without_noise);
		//imwrite("images\\output\\example1Prepo.png", new_image_auto);
		imgSeg(new_image_without_noise, image_segm);
		//namedWindow("Display window", WINDOW_AUTOSIZE);
	//	imshow("Display window", image_segm);
		std::ostringstream name;
		name << "C:\\Users\\micul\\Desktop\\Project1\\images\\output\\" << i << ".png";
		cv::imwrite(name.str(), image_segm);
        cv::waitKey(0); 
    }

	auto end = chrono::steady_clock::now();
	chrono::duration<double> elapsed_seconds = end - start;
	cout << "Proccesing time: " << elapsed_seconds.count() << "s\n";
	return 0;
}




