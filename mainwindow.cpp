#include "mainwindow.h"
#include <functions.h>
#include <QPushButton>
#include "ui_mainwindow.h"
#include <QString>
#include <QLabel>
#include <string>

using namespace std;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent)
{
    QFont font("Microsoft YaHei", 25, 50, true);
    this->setWindowTitle("faceapp");
    this->resize(500,300);
    QLabel *words = new QLabel(this);
    words->setText("welcome to face app!");
    words->setFont(font);
    words->move(100,50);
    words->adjustSize();

    //人脸检测
    this -> detectwindow = new detect;
    detectwindow->hide();
    //人脸录入
    this -> faceinwindow = new facein;
    faceinwindow->hide();
    //人脸识别
    this -> recwindow = new recognize2;
    recwindow->hide();

    //关闭按钮
    QPushButton *close_button = new QPushButton("X",this);
    connect(close_button,&QPushButton::clicked,this,&MainWindow::close);
    close_button->show();
    close_button ->resize(20,20);
    close_button ->move(480,0);

    //最小化按钮
    QPushButton *min_button = new QPushButton("-",this);
    connect(min_button,&QPushButton::clicked,this,&MainWindow::hide);
    min_button->show();
    min_button ->resize(20,20);
    min_button ->move(460,0);

    //人脸检测按钮
    QPushButton *facedet_button = new QPushButton("人脸检测",this);
    connect(facedet_button,&QPushButton::clicked,this,&MainWindow::button_detect);
    connect(detectwindow,&detect::show_mainwindow_detect,this,&MainWindow::detect_signal);
    facedet_button->show();
    facedet_button ->resize(100,50);
    facedet_button ->move(50,200);

    //人脸录入按钮
    QPushButton *facein_button = new QPushButton("人脸录入",this);
    connect(facein_button,&QPushButton::clicked,this,&MainWindow::button_facein);
    connect(faceinwindow,&facein::show_mainwindow_facein,this,&MainWindow::facein_signal);
    facein_button->show();
    facein_button ->resize(100,50);
    facein_button ->move(200,200);

    //人脸识别按钮
    QPushButton *facerec_button = new QPushButton("人脸识别",this);
    connect(facerec_button,&QPushButton::clicked,this,&MainWindow::button_recognize);
    connect(recwindow,&recognize2::show_mainwindow_recognize,this,&MainWindow::recognize_signal);
    facerec_button->show();
    facerec_button ->resize(100,50);
    facerec_button ->move(350,200);
}

//人脸检测窗口
void MainWindow::detect_signal()
{
    this ->show();
    this ->detectwindow -> hide();
}
void MainWindow::button_detect()
{
    this -> hide();
    this ->detectwindow ->show();
}

//人脸录入窗口

void MainWindow::facein_signal()
{
    this ->show();
    this ->faceinwindow -> hide();
}
void MainWindow::button_facein()
{
    this -> hide();
    this -> faceinwindow ->show();
}

//人脸识别窗口
void MainWindow::recognize_signal()
{
    this -> show();
    this -> recwindow ->hide();

}

void MainWindow::button_recognize()
{
    this -> hide();
    this -> recwindow ->show();


}

MainWindow::~MainWindow()
{

}
