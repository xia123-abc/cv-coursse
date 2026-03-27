#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    // 读取图片
    Mat img = imread("text.jpg");
    if (img.empty()) {
        cout << "无法读取图片！" << endl;
        return -1;
    }

    // 输出图像信息
    cout << "尺寸: " << img.cols << "x" << img.rows << endl;
    cout << "通道数: " << img.channels() << endl;
    cout << "数据类型: " << typeToString(img.type()) << endl;

    // 显示原图
    namedWindow("原图", WINDOW_AUTOSIZE);
    imshow("原图", img);

    // 转灰度图
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    namedWindow("灰度图", WINDOW_AUTOSIZE);
    imshow("灰度图", gray);
    imwrite("gray_text.jpg", gray);

    // 像素操作与裁剪
    Vec3b pixel = img.at<Vec3b>(100, 100);
    cout << "像素(100,100) BGR: " << (int)pixel[0] << "," << (int)pixel[1] << "," << (int)pixel[2] << endl;
    Mat crop = img(Rect(0,0,200,200));
    imwrite("crop_text.jpg", crop);
    namedWindow("裁剪图", WINDOW_AUTOSIZE);
    imshow("裁剪图", crop);

    waitKey(0);
    destroyAllWindows();
    return 0;
}
