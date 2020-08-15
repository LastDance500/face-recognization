#ifndef DETECT_H
#define DETECT_H

#include <QWidget>
#include <QtMultimedia/QMediaPlayer>
#include <QtMultimedia/QMediaPlaylist>
#include <QtMultimedia/qmediaplayercontrol.h>
#include <QtMultimediaWidgets/QVideoWidget>
#include <opencv2/core.hpp>

class detect : public QWidget
{
    Q_OBJECT
public:
    explicit detect(QWidget *parent = nullptr);

signals:
    //信号没有返回值，可以有参数
    void show_mainwindow_detect();

public slots:
    void emit_detectsignal();
    void facedetect();
    QImage Mat2QImage(const cv::Mat &mat);
private:

};

#endif // DETECT_H
