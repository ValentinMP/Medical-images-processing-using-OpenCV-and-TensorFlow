#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QtWidgets>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <math.h>
#include <windows.h>
#include <algorithm>
#include <sstream> //for image parsing
#include <time.h> // for execution time measurment
#include <QtCore/QCoreApplication>
#include <QFileDialog>
#include <QDebug>
using namespace cv;
using namespace std;

void CorectieGamma(Mat& src, Mat& dst, float parametru_luminozitate)

{

    unsigned char tabel_lut[256];

    for (int i = 0; i < 256; i++)

    {

        tabel_lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), parametru_luminozitate) * 255.0f);

    }

    dst = src.clone();

    const int canale = dst.channels();

    switch (canale)

    {

    case 1:

    {

        MatIterator_<uchar> it, end;

        for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)

            *it = tabel_lut[(*it)];

        break;

    }

    case 3:

    {

        MatIterator_<Vec3b> it, end;

        for (it = dst.begin<Vec3b>(), end = dst.end<Vec3b>(); it != end; it++)

        {

            (*it)[0] = tabel_lut[((*it)[0])];

            (*it)[1] = tabel_lut[((*it)[1])];

            (*it)[2] = tabel_lut[((*it)[2])];

        }

        break;

    }

    }

}


void AjustareAutomataContrastLuminozitate(Mat& src, Mat& dst, float procent_taiere)
{

    CV_Assert(procent_taiere >= 0);
    CV_Assert((src.type() == CV_8UC1) || (src.type() == CV_8UC3) || (src.type() == CV_8UC4));

    int histSize = 256;
    float alfa, beta;
    double minGray = 0, maxGray = 0;

    //calculare histograma
    cv::Mat gray;
    if (src.type() == CV_8UC1) gray = src;
    else if (src.type() == CV_8UC3) cvtColor(src, gray, cv::COLOR_RGB2GRAY);
    else if (src.type() == CV_8UC4) cvtColor(src, gray, cv::COLOR_RGB2GRAY);
    if (procent_taiere == 0)
    {
        //se calculeaza pe intreg domeniul
        cv::minMaxLoc(gray, &minGray, &maxGray);
    }
    else
    {
        cv::Mat hist; //histograma

        float range[] = { 0, 256 };
        const float* histRange = { range };
        bool uniform = true;
        bool accumulate = false;
        calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

       //calcularea distributiei cumulative din histograma
        std::vector<float> accumulator(histSize);
        accumulator[0] = hist.at<float>(0);
        for (int i = 1; i < histSize; i++)
        {
            accumulator[i] = accumulator[i - 1] + hist.at<float>(i);
        }

        //localizarea punctelor de taiere la valoarea ceruta
        float max = accumulator.back();
        procent_taiere *= (max / 100.0); //transformarea valorii de taiere in procente
        procent_taiere /= 2.0; // taierea la stanga si la drepta
        //cautarea valorii de taiere pentru stanga
        minGray = 0;
        while (accumulator[minGray] < procent_taiere)
            minGray++;

        //cautarea valorii de taiaere pentru dreapta
        maxGray = histSize - 1;
        while (accumulator[maxGray] >= (max - procent_taiere))
            maxGray--;
    }

    //domeniul actual al histogramei, dupa localizarea punctelor de taiere
    float inputRange = maxGray - minGray;

    alfa = (histSize - 1) / inputRange;  //alfa extinde domeniul curent la intreg domeniul al histogramei
    beta = -minGray * alfa;   // beta modifica domeniul actual astfel incat valoarea taiereii din stanga sa fie 0           

   
   // Aplicarea normalizarii luminozitatii si a contrastului
    // Iesire = Intrare * alfa + beta;
    src.convertTo(dst, -1, alfa, beta);

   // In cazul imaginilor cu transparenta, pentru canalul afla
    if (dst.type() == CV_8UC4)
    {
        int from_to[] = { 3, 3 };
        cv::mixChannels(&src, 4, &dst, 1, from_to, 1);
    }
    return;
}




float variatieClasa(int hist[], int nivel, float val, int valoare_pixel)
{
    long long total = valoare_pixel * val;
    int n = 0;
    long long m = 0;
    for (int i = 0; i < nivel; i++)
    {
        m += i * hist[i];
        n += hist[i];
    }
    long long rest = total - m;
    int pixeli_ramasi = valoare_pixel - n;
    float proportie_pixeli_fundal0 = (1.0 * n) / (1.0 * valoare_pixel);
    float media_pixelilor_fundal1 = (1.0 * rest) / (1.0 * valoare_pixel);
    float proportie_pixeli_primPlan0 = (1.0 * m) / (1.0 * n);
    float media_pixelilor_primPlan1 = (1.0 * rest) / (1.0 * pixeli_ramasi);
    return proportie_pixeli_fundal0 * media_pixelilor_fundal1 * (proportie_pixeli_primPlan0 - media_pixelilor_primPlan1) * (proportie_pixeli_primPlan0 - media_pixelilor_primPlan1);
}
void segmentareImagine(Mat& input_image, Mat& output_image)
{
    //imwrite("C:\\Users\\Vali\\Desktop\\ostsu.png", input_image);

    long long u = 0;
    int hist[256];
    for (int i = 0; i < 256; i++)
        hist[i] = 0;
    //int sz = input_image.cols * input_image.rows;
    for (int i = 0; i < input_image.rows; i++)
    {
        for (int j = 0; j < input_image.cols; j++)
        {
            int n = input_image.at<uchar>(i, j);
            u += n;
            hist[n]++;
        }
    }
    int valoare_pixel = input_image.rows * input_image.cols;
    float val = (1.0 * u) / float(valoare_pixel);
    float max = 0;
    int threshold = 0;
    for (int i = 1; i < 255; i++)
    {
        int x = variatieClasa(hist, i, val, valoare_pixel);
        if (x > max)
        {
            max = x;
            threshold = i;
        }
    }
    for (int i = 0; i < input_image.rows; i++)
    {
        for (int j = 0; j < input_image.cols; j++)
        {
            if (input_image.at<uchar>(i, j) > threshold)
            {
                input_image.at<uchar>(i, j) = 255;
            }
            else
                input_image.at<uchar>(i, j) = 0;
        }
    }



    output_image = input_image.clone();

}


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    // Create the button, make "this" the parent
       m_button = new QPushButton("Load image", this);
       // set size and location of the button
       m_button->setGeometry(QRect(QPoint(100, 100),
       QSize(200, 50)));

       // Connect button signal to appropriate slot
       connect(m_button, SIGNAL (released()), this, SLOT (on_pushButton_clicked()));



       // Create the button, make "this" the parent
          second_button = new QPushButton("Process", this);
          // set size and location of the button
          second_button->setGeometry(QRect(QPoint(100, 150),
          QSize(200, 50)));

          // Connect button signal to appropriate slot
          connect(second_button, SIGNAL (released()), this, SLOT (on_pushButton_clicked2()));
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_pushButton_clicked()
{

    QString fileName = QFileDialog::getOpenFileName(this,
            tr("Open Image"), ".",
          tr("Image Files (*.png *.jpg *.jpeg *.bmp)"));

       image= imread(fileName.toLatin1().data());
       namedWindow("Original Image");
       imshow("Original Image", image);


}




void MainWindow::on_pushButton_clicked2()
{
    if (image.empty())                      // Check for invalid input
        {
            cout << "Could not open or find the image" << std::endl;
           exit(1);
        }

        cvtColor(image, grayscale_image, COLOR_BGR2GRAY);

    CorectieGamma(grayscale_image, new_image_gamma_corr, 0.5);
     imwrite("C:\\Users\\Vali\\Desktop\\Prezentare\\Imagini_intermediare\\Gamma_corretion.png", new_image_gamma_corr);
        AjustareAutomataContrastLuminozitate(new_image_gamma_corr, new_image_auto, 4);
         imwrite("C:\\Users\\Vali\\Desktop\\Prezentare\\Imagini_intermediare\\brightness_auto.png", new_image_auto);
        //imwrite("images\\output\\example1_AutoBrightnessAndContrast.png", new_image_auto);
        //BrightnessAndContrastAuto(new_image_gamma_corr, image_just_with_bright, 4);

        //medianFilter(new_image_auto, new_image_without_noise);
        bilateralFilter(new_image_auto, new_image_without_noise, 15, 80, 80);
         imwrite("C:\\Users\\Vali\\Desktop\\Prezentare\\Imagini_intermediare\\img_without_noise.png", new_image_without_noise);
        //new_image_without_noise = bilateralFilterOwn(new_image_auto, 15, 80, 80);
        //imwrite("images\\output\\example1_bilateralFilterOwn.png", new_image_without_noise);
        //imwrite("images\\output\\example1Prepo.png", new_image_auto);
         segmentareImagine(new_image_without_noise, image_segm);

        imwrite("C:\\Users\\Vali\\Documents\\TensorFlow\\models\\research\\object_detection\\test_images\\image1.png", image_segm);
   // cv::threshold(new_image_without_noise, image_segm, 0, 255, THRESH_BINARY | THRESH_OTSU);
        namedWindow("After");
        imshow("After", image_segm);


}
