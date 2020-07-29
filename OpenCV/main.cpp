#include "mainwindow.h"

#include <QApplication>

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

using namespace cv;
using namespace std;


int main(int argc, char *argv[])
{

    QApplication a(argc, argv);
    MainWindow w;
    w.show();



    return a.exec();
}
