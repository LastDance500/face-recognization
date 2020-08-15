#ifndef RECOGNIZE2_H
#define RECOGNIZE2_H

#include <QWidget>
#include <QLabel>
#include <QTimer>
#include <QPushButton>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/face.hpp>
#include <opencv2/face/facerec.hpp>


using namespace cv;
using namespace std;
using namespace cv::face;

namespace Ui {
class recognize2;
}

class recognize2 : public QWidget
{
    Q_OBJECT

public:
    explicit recognize2(QWidget *parent = 0);
    recognize2 *recwindow;
    ~recognize2();


signals:
    void show_mainwindow_recognize();

private:
    Ui::recognize2 *ui;
    cv::Mat frame;
    cv::VideoCapture capture;
    QImage  image;
    QTimer *timer;
    double rate;
    CascadeClassifier cascade;



public slots:
    void emit_recognizesignal();
    void nextFrame();

private slots:
    void on_back_button_clicked();
    void on_play_button_clicked();

};

#endif // RECOGNIZE2_H
