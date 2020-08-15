#include "ui_recognize2.h"
#include "recognize2.h"

#include <QPushButton>
#include <QWidget>
#include <QLabel>
#include <QTimer>
#include <QImage>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/face.hpp>
#include <opencv2/face/facerec.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml.hpp>

#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>


static Ptr<FaceRecognizer> modelPCA = EigenFaceRecognizer::create();

using namespace std;
using namespace cv;
using namespace cv::face;

recognize2::recognize2(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::recognize2)
{
    cascade.load("/home/tbs/opencv2/opencv-4.3.0/data/haarcascades/haarcascade_frontalface_alt2.xml");
    modelPCA->read("/home/tbs/test/opencv_project/MyFacePCAModel.xml");
    ui->setupUi(this);
    //connect(ui->play_button,&QPushButton::clicked,this,&recognize2::readFrame);
    connect(ui->back_button,&QPushButton::clicked,this,&recognize2::emit_recognizesignal);

}

recognize2::~recognize2()
{
    delete ui;
}

void recognize2::nextFrame()
{
    capture >> frame;
    Mat gray;
    //训练好的文件名称，放置在可执行文件同目录下
    //建立用于存放人脸的向量容器
    vector<Rect> faces(0);
    cvtColor(frame, gray, CV_BGR2GRAY);
    //改变图像大小，使用双线性差值
    //resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
    //变换后的图像进行直方图均值化处理
    equalizeHist(gray, gray);
    cascade.detectMultiScale(gray, faces,  1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    Mat face;
    Point text_lb;
    for (size_t i = 0; i < faces.size(); i++)
    {
        if (faces[i].height > 0 && faces[i].width > 0)
        {
            face = gray(faces[i]);
            text_lb = Point(faces[i].x, faces[i].y);
            rectangle(frame, faces[i], Scalar(255, 0, 0), 1, 8, 0);
        }
    }

    Mat face_test;
    int predictPCA = 0;

    //使用csv文件去读图像和标签
    for(size_t i =0;i<faces.size();i++)
    {
        if (face.rows >= 120){
            cv::resize(face, face_test, Size(92, 112));
        }
        if (!face_test.empty()){
            //测试图像应该是灰度图
            predictPCA = modelPCA->predict(face_test);
        }

        //cout << predictPCA << endl;
        string name;
        int row=0;
        fstream file;
        file.open("/home/tbs/test/facedata/ORL_Faces/name.txt",ios::in);
        while(getline(file,name))
        {
            if(row == predictPCA-40)
               break;
            else
               row++;
        }
        file.close();
        putText(frame, name, text_lb, FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));
    }
    cv::resize(frame,frame,Size(300,300));
    //Mat文件转换为QImage
    QImage qImg;
    if(frame.channels()==3)                             //3 channels color image
    {

        cv::cvtColor(frame,frame,CV_BGR2RGB);
        qImg =QImage((const unsigned char*)(frame.data),
                    frame.cols, frame.rows,
                    frame.cols*frame.channels(),
                    QImage::Format_RGB888);
    }
    else if(frame.channels()==1)                    //grayscale image
    {
        qImg =QImage((const unsigned char*)(frame.data),
                    frame.cols,frame.rows,
                    frame.cols*frame.channels(),
                    QImage::Format_Indexed8);
    }
    else
    {
        qImg =QImage((const unsigned char*)(frame.data),
                    frame.cols,frame.rows,
                    frame.cols*frame.channels(),
                    QImage::Format_RGB888);
    }
    image = qImg;

    //cv::imshow("face",frame);
    ui->img_label->setPixmap(QPixmap::fromImage(image));
}

void recognize2::emit_recognizesignal()
{

    emit show_mainwindow_recognize();

}

void recognize2::on_back_button_clicked()
{

}

void recognize2::on_play_button_clicked()
{
    capture.open(0);           //open the default camera
    if (capture.isOpened())
    {
        rate= capture.get(CAP_PROP_FPS);
        timer = new QTimer(this);
        timer->setInterval(1000/rate);   //set timer match with FPS
        connect(timer, SIGNAL(timeout()), this, SLOT(nextFrame()));
        timer->start();
        //}
    }
}



