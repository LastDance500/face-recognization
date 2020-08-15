#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QWidget>
#include <QPushButton>
#include "detect.h"
#include "facein.h"
#include "recognize2.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    detect *detectwindow;
    facein *faceinwindow;
    recognize2 *recwindow;

//槽函数
public slots:
    //人脸检测部分
    void button_detect();
    void detect_signal();
    //人脸录入部分
    void button_facein();
    void facein_signal();
    //人脸识别部分
    void button_recognize();
    void recognize_signal();


private slots:


private:
     Ui::MainWindow* ui;

};


#endif // MAINWINDOW_H
