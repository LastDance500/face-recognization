#include "detect.h"
#include <QMainWindow>
#include <QWidget>
#include <QPushButton>
#include "ui_detect.h"
#include <QFileDialog>
#include <QDebug>
#include <QLabel>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/face.hpp>
#include <opencv2/face/facerec.hpp>
#include <opencv2/imgproc/types_c.h>

#include <iostream>
#include <stdio.h>
#include <ctime>
#include <stdio.h>
#include <string>
#include <vector>
#include <math.h>

using namespace std;
using namespace cv;
using namespace dnn;

detect::detect(QWidget *parent) : QWidget(parent)
{

    //人脸检测界面
    this->setWindowTitle("人脸检测");
    this->resize(600,400);
    //回退button
    QPushButton *back_button = new QPushButton("back",this);
    connect(back_button,&QPushButton::clicked,this,&detect::emit_detectsignal);
    back_button->show();
    back_button ->resize(40,20);
    back_button ->move(10,10);
    //开始播放button
    QPushButton *play_button = new QPushButton("play",this);
    connect(play_button,&QPushButton::clicked,this,&detect::facedetect);
    play_button->show();
    play_button ->resize(40,20);
    play_button ->move(100,100);

}

QImage detect::Mat2QImage(const cv::Mat &mat)
{
    QImage img;
    Mat rgb;
    if(mat.channels()==3)
    {
        //cvt Mat BGR 2 QImage RGB
        cvtColor(mat,rgb,CV_BGR2RGB);
        img =QImage((const unsigned char*)(rgb.data),
                    rgb.cols,rgb.rows,
                    rgb.cols*rgb.channels(),
                    QImage::Format_RGB888);
    }
    else
    {
        img =QImage((const unsigned char*)(mat.data),
                    mat.cols,mat.rows,
                    mat.cols*mat.channels(),
                    QImage::Format_RGB888);
    }
    return img;
}

void detect::facedetect()
{
    //摄像头播放
    string modelConfiguration ="/home/tbs/test/opencv_project/deploy.prototxt";
    string modelBinary = "/home/tbs/test/opencv_project/res10_300x300_ssd_iter_140000_fp16.caffemodel";

    dnn::Net net = readNetFromCaffe(modelConfiguration, modelBinary);//从caffe中读出模型

    VideoCapture cap(-1);//打开摄像头,windows下是0，Linux下是-1
    //VideoCapture cap(""); //或者是视频路径

    if(!cap.isOpened())
    {
        cout <<"could not open the camera"<<endl;
    }

//    //存储视频
//    //截取一帧，提取一些信息
//    Mat oneframe;
//    cap >> oneframe;
//    string outpuVideopath = "";    //输出路径
//    VideoWriter writer;
//    int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
//    double fps = 4.5;
//    cv::resize(oneframe,oneframe,Size(300,300));
//    writer.open(outpuVideopath,codec,fps,oneframe.size(),true);
//    if(!writer.isOpened())
//    {
//        cerr << "coule not open the output video file to write.\n";
//    }



    const size_t inWidth = 300;
    const size_t inHeight = 300;
    const double inScaleFactor = 1.0;
    const Scalar meanVal(104.0, 117.0, 123.0);

    //开始播放图像
    while(true)
    {
        Mat frame;
        cap >> frame;   //从摄像头或者视频中获取帧图像

        if(frame.empty())    //没提取到就退出
        {
            break;
        }

        if(frame.channels()==4)     //
            cvtColor(frame,frame,COLOR_BGRA2BGR);

        //cvtColor(frame,frame,COLOR_BGR2GRAY);//转为灰度图
        //将输入图像转化为Net可识别的blob格式
        Mat inputBlob = blobFromImage(frame, inScaleFactor,
            Size(inWidth, inHeight), meanVal, false, false);
        //输入
        net.setInput(inputBlob,"data");
        //forwards计算输出
        Mat detection = net.forward("detection_out");

        //vector<double> layersTimings;
        //double freq = getTickFrequency() /1000;    //用于返回CPU的频率，里面的单位是秒，也就是一秒内重复的次数
        //double time1 = net.getPerfProfile(layersTimings)/freq;   //

        Mat detectionMat(detection.size[2],detection.size[3], CV_32F, detection.ptr<float>());

        //ostringstream ss;
        //ss<<"FPS:" << 1000/time1<<";time"<<time1<<"ms";
        //putText(frame, ss.str(), Point(20, 20), 0, 0.8, Scalar(0, 0, 255),2);

        float min_confidence = 0.5;
        float confidenceThreshold = min_confidence;
        int faceNum = 0;
        for (int i = 0; i< detectionMat.rows;i++)
        {
            float confidence = detectionMat.at<float>(i,2);

            if(confidence > confidenceThreshold)
            {
                int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

                Rect object((int)xLeftBottom, (int)yLeftBottom,(int)(xRightTop - xLeftBottom),
                            (int)(yRightTop - yLeftBottom));

                rectangle(frame,object,Scalar(0,255,0));
                //ss.str("");
                //ss<<confidence;
                //String conf(ss.str());
                //String label = "Face:" +conf;

                //int baseLine = 0;
                //Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                ++faceNum;
            }
        }

        //输出人脸的数量
        ostringstream faceNum1;
        faceNum1 << "FaceNum: " << faceNum;
        putText(frame, faceNum1.str(), Point(25, 40), 0, 0.8, Scalar(0, 0, 255), 2);
        faceNum = 0;


        //写入视频
    //        Mat newframe;
    //        newframe = frame;
    //        cv::resize(newframe,newframe,Size(300,300));
    //        writer << newframe;

        //界面
        QImage image = QImage((const uchar*)frame.data,frame.cols,frame.rows,QImage::Format_RGB888).rgbSwapped();
        QLabel *label= new QLabel(this);
        label->move(100,50);
        label->setPixmap(QPixmap::fromImage(image));
        label->adjustSize();

        //cv::imshow("Camera Play", frame);
//        if(waitKey(20)==27)
//        {
//            break;
//        }
    }

}

void detect::emit_detectsignal()
{

    emit show_mainwindow_detect();

}
