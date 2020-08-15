#ifndef RECOGNIZE_H
#define RECOGNIZE_H

#include <QWidget>
#include <opencv2/core.hpp>

namespace Ui
{
    class recognizeUi;
}


class recognize : public QWidget
{
    Q_OBJECT
public:
    explicit recognize(QWidget *parent = nullptr);
    ~recognize();

signals:
    void show_mainwindow_recognize();


public slots:
    void emit_recognizesignal();
    void facerec1();
    QImage Mat2QImage(const cv::Mat &mat);

private:
    Ui::recognizeUi *rec;

};

#endif // RECOGNIZE_H
