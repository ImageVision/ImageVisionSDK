#include "CImg.h"
#include <iostream>
using namespace std;
using namespace cimg_library;
#include "ext.h"

//���ָ�ʽ��ת
int main() 
{
	CImg<unsigned char> img;
	img = readImage("input.jpg");  //֧�ֶ��ָ�ʽ��input.bmp��input.png��input.jpg��input.tga 
	saveImage("output.bmp", img);   //֧�ֶ��ָ�ʽ��output.bmp��output.png��output.jpg��output.tga
	return 0;
}
