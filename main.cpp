#include "mainwindow.h"
#include <QApplication>
//qt头文件没有.h后缀
//qt一个类对应一个头文件，类名就是头文件名
//QApplication 应用程序类

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    return a.exec();  //程序进入消息循环，等待对用户输入进行响应
}
