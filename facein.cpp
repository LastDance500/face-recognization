#include <QPushButton>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>
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
#include "facein.h"
#include "ui_facein.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace cv::face;

facein::facein(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::facein)
{
    ui->setupUi(this);

}

//获取文件夹的数量
int facein::get_file_count(char *root)
{
    DIR *dir;
    struct dirent * ptr;
    int total = 0;
    char path[50];
    dir = opendir(root); /* 打开目录*/
    if(dir == NULL)
    {
        perror("fail to open dir");
        exit(1);
    }
    errno = 0;
    while((ptr = readdir(dir)) != NULL)
    {
        //顺序读bai取每一个目录项；
        //跳过“..”和“du.”两个目录
        if(strcmp(ptr->d_name,".") == 0 || strcmp(ptr->d_name,"..") == 0)
        {
        continue;
        }
        //如果是目录，则递归调用dao get_file_count函数
        if(ptr->d_type == DT_DIR)
        {
            //sprintf(path,"%s%s/",root,ptr->d_name);
            //printf("%s/n",path);
            total++;
            //total += get_file_count(path);
        }
    }
    if(errno != 0)
    {
        printf("fail to read dir"); //失败则输出提示信息
        exit(1);
    }
    closedir(dir);
    return total;
}

//采集人脸,处理所采集的数据
void facein::facecollect1()
{
    //加载分类器
    CascadeClassifier cascade;
    cascade.load("/home/tbs/opencv2/opencv-4.3.0/data/haarcascades/haarcascade_frontalface_alt2.xml");
    //拍照程序
    VideoCapture cap;
    cap.open(0);

    Mat frame;
    int pic_num = 1;

    QList<QLabel *> labels =ui->widget->findChildren<QLabel *>();

    while (1)
    {
        cap >> frame;
        //定义容器存储faces，rect是矩形对象
        vector<Rect> faces;
        Mat frame_gray;
        //输出灰度图像
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        //检测出人脸，存放在faces容器中
        cascade.detectMultiScale(frame_gray, faces, 1.1, 4, 0, Size(100, 100), Size(500, 500));
        //1.1表示每次图像尺寸减少的比例
        //4表示每一个目标至少要被检测到3次才算是真的目标(因为周围的像素和不同的窗口大
        //小都可以检测到人脸表示每一个目标至少要被检测到3次才算是真的目标(因为周围的像素和不同的窗口大小都可以检测到人脸

        //画矩形在人脸上
        for (size_t i = 0; i < faces.size(); i++)
        {
            rectangle(frame, faces[i], Scalar(255, 0, 0), 2, 8, 0);
        }

        //处理人脸的大小，并存储在文件夹中
        if (faces.size() == 1)
        {
            Mat faceROI = frame_gray(faces[0]);
            Mat myFace;
            cv::resize(faceROI, myFace, Size(92, 112));//数据集中图像的大小
            putText(frame, to_string(pic_num), faces[0].tl(),3, 1.2, (0, 0, 255), 2, LINE_AA);
            //自己创建文件夹，存储在该文件夹下
            int facefilenum ;
            facefilenum = get_file_count("/home/tbs/test/facedata/ORL_Faces/");
            string filename = format("/home/tbs/test/facedata/ORL_Faces/s%d/%d.jpg",facefilenum,pic_num);
            imwrite(filename, myFace);
            frame = myFace;

            if(pic_num<=9)
            {
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
                QImage image = qImg;

                labels.at(pic_num-1)->setPixmap(QPixmap::fromImage(image));
            }
            waitKey(500);
            sleep(1);
            //destroyWindow(filename);
            pic_num++;
            //默认拍十张照片，到第十一张的时候停止
            if (pic_num == 11)
            {
                return ;
            }
        }
        //imshow("frame", frame);
        waitKey(100);
    }
    return ;
}

//生成csv文件，列出一个目录下所有文件
void facein::scan_one_dir1( const char * dir_name, int dirnum, const char * dir_txt)
{

    if( NULL == dir_name )
    {
        cout<<" dir_name is null ! "<<endl;
        return;
    }

    struct stat s;
    lstat( dir_name , &s );//读取dir_name下的文件，存储在stat结构体中

    if(! S_ISDIR( s.st_mode ))    //读取权限
    {
        return;
    }

    struct dirent * filename;    //dirent结构体
    DIR * dir;                   //DIR结构体
    dir = opendir( dir_name );
    if( NULL == dir )
    {
        return;
    }

    ofstream outfile;
    outfile.open(dir_txt,ios::app);
    while( ( filename = readdir(dir) ) != NULL )
    {
        if( strcmp( filename->d_name , "." ) == 0 ||
            strcmp( filename->d_name , "..") == 0)
            continue;
        char wholePath[128] = {0};
        sprintf(wholePath, "%s/%s;%d", dir_name, filename->d_name,dirnum);
        outfile << wholePath <<endl;
        //cout << wholePath << endl;
    }
    outfile.close();
}

//获得自己的at.txt文件
int facein::getcsv1()
{
    int dirnum = 1;
    remove("/home/tbs/test/facedata/ORL_Faces/at.txt");
    while(1)
    {
        char const dir_txt[] = ("/home/tbs/test/facedata/ORL_Faces/at.txt");
        char str[128] ={0};
        char str_temp[] = ("/home/tbs/test/facedata/ORL_Faces/s");
        sprintf(str,"%s%d",str_temp,dirnum);
        const char *ptr = str;
        if(access(ptr,0))
        {
            break;
        }
        scan_one_dir1(ptr,dirnum,dir_txt);
        //recursion_scan_dir_file(str, 1);
        dirnum++;
    }
    return 0;
}

//将名字存入txt文件
void facein::namein()
{
    QString s= ui->lineEdit->text();
    std::string new_s = s.toStdString();
    ofstream os;     //创建一个文件输出流对象
    os.open("/home/tbs/test/facedata/ORL_Faces/name.txt",ios::app);
    os << new_s<<endl;
    os.close();
}

//使用csv文件去读图像和标签，主要使用stringstream和getline方法
void facein::read_csv1(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';')
{
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file)
    {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);;
    }
    string line, path, classlabel;
    while (getline(file, line))
    {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if (!path.empty() && !classlabel.empty())
        {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

//训练自己的脸部模型
void facein::facetrain1()
{
    string fn_csv = "/home/tbs/test/facedata/ORL_Faces/at.txt";
    vector<Mat> images;
    vector<int> labels;

    //查看是否能读取到csv文件
    try
    {
        read_csv1(fn_csv, images, labels,';');
    }
    catch(cv::Exception& e)
    {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        exit(1);
    }

    //排除最后一张照片，作为测试照片
    Mat testSample = images[images.size() - 1];
    int testLabel = labels[labels.size() - 1];
    images.pop_back();
    labels.pop_back();

    //训练自己的模型
    Ptr<BasicFaceRecognizer> model = EigenFaceRecognizer::create();
    model->train(images, labels);
    model->save("/home/tbs/test/opencv_project/MyFacePCAModel.xml");

    Ptr<BasicFaceRecognizer> model1 = FisherFaceRecognizer::create();
    model1->train(images, labels);
    model1->save("/home/tbs/test/opencv_project/MyFaceFisherModel.xml");

    Ptr<LBPHFaceRecognizer> model2 = LBPHFaceRecognizer::create();
    model2->train(images, labels);
    model2->save("/home/tbs/test/opencv_project/MyFaceLBPHModel.xml");

    //加载分类器
    int predictedLabel = model->predict(testSample);//加载分类器
    int predictedLabel1 = model1->predict(testSample);
    int predictedLabel2 = model2->predict(testSample);

    string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
    string result_message1 = format("Predicted class = %d / Actual class = %d.", predictedLabel1, testLabel);
    string result_message2 = format("Predicted class = %d / Actual class = %d.", predictedLabel2, testLabel);
    cout << result_message << endl;
    cout << result_message1 << endl;
    cout << result_message2 << endl;

    getchar();
    //waitKey(0);

}

void facein::emit_faceinsignal()
{

        emit show_mainwindow_facein();

}

facein::~facein()
{
    delete ui;
}

void facein::on_collect_button_clicked()
{
   facecollect1();
   getcsv1();
   namein();
}

void facein::on_back_button_clicked()
{
    connect(ui->back_button,&QPushButton::clicked,this,&facein::emit_faceinsignal);
}

void facein::on_train_button_clicked()
{
    facetrain1();
}
