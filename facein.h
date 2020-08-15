#ifndef FACEIN_H
#define FACEIN_H

#include <QWidget>
#include <opencv2/core.hpp>

#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <string>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>

using namespace std;
using namespace cv;

namespace Ui {
class facein;
}
class facein : public QWidget
{
    Q_OBJECT
public:
    explicit facein(QWidget *parent = nullptr);
    ~facein();

signals:
    void show_mainwindow_facein();


private:
    Ui::facein *ui;
    void facecollect1();
    void scan_one_dir1( const char * dir_name, int dirnum, const char * dir_txt);
    int getcsv1();
    //static Mat norm_0_255(InputArray _src);
    void read_csv1(const string&, vector<Mat>&, vector<int>&, char);
    void facetrain1();
    void facein1();
    int get_file_count(char *root);
    void namein();

public slots:
    void emit_faceinsignal();

private slots:
    void on_collect_button_clicked();
    void on_back_button_clicked();
    void on_train_button_clicked();
};

#endif // FACEIN_H
