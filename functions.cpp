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

using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace cv::face;

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const Scalar meanVal(104.0, 117.0, 123.0);
int filenum = 0;

//获取时间
string getTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp,sizeof(tmp),"%Y-%m-%d %H:%M:%s", localtime(&timep));
    return tmp;
}

//摄像头检测人脸
int cameraWindow()
{
    string modelConfiguration ="/home/tbs/test/opencv_project/deploy.prototxt";
    string modelBinary = "/home/tbs/test/opencv_project/res10_300x300_ssd_iter_140000_fp16.caffemodel";

    dnn::Net net = readNetFromCaffe(modelConfiguration, modelBinary);//从caffe中读出模型

    VideoCapture cap(-1);//打开摄像头,windows下是0，Linux下是-1
    //VideoCapture cap(""); //或者是视频路径


    if(!cap.isOpened())
    {
        cout <<"could not open the camera"<<endl;
        return -1;
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

        vector<double> layersTimings;
        double freq = getTickFrequency() /1000;    //用于返回CPU的频率，里面的单位是秒，也就是一秒内重复的次数
        double time1 = net.getPerfProfile(layersTimings)/freq;   //

        Mat detectionMat(detection.size[2],detection.size[3], CV_32F, detection.ptr<float>());

        ostringstream ss;
        ss<<"FPS:" << 1000/time1<<";time"<<time1<<"ms";
        putText(frame, ss.str(), Point(20, 20), 0, 0.8, Scalar(0, 0, 255),2);

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
                ss.str("");
                ss<<confidence;
                String conf(ss.str());
                String label = "Face:" +conf;

                int baseLine = 0;
                Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
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

        cv::imshow("Camera Play", frame);
        if(waitKey(20)==27)
        {
            break;
        }

    }
    return 0;
}

//采集人脸,处理所采集的数据
void facecollect()
{
    //加载分类器
    CascadeClassifier cascade;
    cascade.load("/home/tbs/opencv2/opencv-4.3.0/data/haarcascades/haarcascade_frontalface_alt2.xml");
    //拍照程序
    VideoCapture cap;
    cap.open(0);

    Mat frame;
    int pic_num = 1;
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
            resize(faceROI, myFace, Size(92, 112));//数据集大小
            putText(frame, to_string(pic_num), faces[0].tl(),3, 1.2, (0, 0, 255), 2, LINE_AA);
            //自己创建文件夹，存储在该文件夹下
            string filename = format("/home/tbs/test/facedata/face%d.jpg", pic_num);
            imwrite(filename, myFace);
            imshow(filename, myFace);
            waitKey(500);
            destroyWindow(filename);
            pic_num++;
            //默认拍十张照片，到第十一张的时候停止
            if (pic_num == 11)
            {
                return ;
            }
        }
        imshow("frame", frame);
        waitKey(100);
    }
    return ;
}

//生成csv文件，列出一个目录下所有文件
void scan_one_dir( const char * dir_name, int dirnum, const char * dir_txt)
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
        cout << wholePath << endl;
    }
    outfile.close();
}

//获得自己的csv文件
int getcsv()
{

    int dirnum = 1;
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
        scan_one_dir(ptr,dirnum,dir_txt);
        //recursion_scan_dir_file(str, 1);
        dirnum++;
    }
    return 0;
}

//建立自己的人脸模型，read_csv和facetrain
static Mat norm_0_255(InputArray _src)
{
    Mat src = _src.getMat();
    // 创建和返回一个归一化后的图像矩阵:
    Mat dst;
    switch (src.channels()) {
    case 1:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

//使用csv文件去读图像和标签，主要使用stringstream和getline方法
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';')
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
void facetrain()
{
    string fn_csv = "/home/tbs/test/facedata/ORL_Faces/at.txt";

    vector<Mat> images;
    vector<int> labels;

    //查看是否能读取到csv文件
    try
    {
        read_csv(fn_csv, images, labels);
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

//调用摄像头检测人脸
void facerec()
{

        VideoCapture cap(0);    //打开默认摄像头
        if (!cap.isOpened())
        {
            return ;
        }
        Mat frame;
        Mat gray;

        CascadeClassifier cascade;
        bool stop = false;
        //训练好的文件名称，放置在可执行文件同目录下
        cascade.load("/home/tbs/opencv2/opencv-4.3.0/data/haarcascades/haarcascade_frontalface_alt2.xml");

        Ptr<FaceRecognizer> modelPCA = EigenFaceRecognizer::create();
        modelPCA->read("/home/tbs/test/opencv_project/MyFacePCAModel.xml");
        while (!stop)
        {
            cap >> frame;

            //建立用于存放人脸的向量容器
            vector<Rect> faces(0);
            cvtColor(frame, gray, CV_BGR2GRAY);
            //改变图像大小，使用双线性差值
            //resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
            //变换后的图像进行直方图均值化处理
            equalizeHist(gray, gray);
            cascade.detectMultiScale(gray, faces,
                1.1, 2, 0
                //|CV_HAAR_FIND_BIGGEST_OBJECT
                //|CV_HAAR_DO_ROUGH_SEARCH
                | CASCADE_SCALE_IMAGE,
                Size(30, 30));

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
            if (face.rows >= 120)
            {
                resize(face, face_test, Size(92, 112));

            }

            if (!face_test.empty())
            {
                //测试图像应该是灰度图
                predictPCA = modelPCA->predict(face_test);
            }

            cout << predictPCA << endl;
            if (predictPCA == 41)
            {
                string name = "XiaoZhang";
                putText(frame, name, text_lb, FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));
            }

            imshow("face", frame);
            if (waitKey(50) >= 0)
                stop = true;
        }

        return ;
}
