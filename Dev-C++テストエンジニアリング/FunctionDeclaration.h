#include<string>
#include <iostream>
#include <list>
#include<vector>
#include <windows.h>
using namespace std;

namespace ezsift {
#define DEGREE (128)   //SIFT�ؼ���: 128ά
struct SiftKeypoint {
    int octave;   // octave����
    int layer;    // layer����
    float rlayer; // layerʵ������
    float r;     // ��һ����row����
    float c;     // ��һ����col����
    float scale; // ��һ����scale
    float ri;          // row����(layer)
    float ci;          // column����(layer)
    float layer_scale; // scale(layer)
    float ori; // ����(degrees)
    float mag; // ģֵ
    float descriptors[DEGREE]; //������ 
};
}

typedef struct 
{
	unsigned char red, green, blue;		//���ص���ɫ��RGB����/��/������ʾ
} PPMPixel;
typedef struct 
{
	unsigned int width, height;			// ͼ��Ŀ�Ⱥ͸߶ȣ�������Ϊ��λ��
	PPMPixel *data;						// ����ͼ�������
} PPMImage;

typedef struct {
    int x;
    int y;
    unsigned char red;
    unsigned char green;
    unsigned char blue;
}SobelImage;

typedef struct rgb_packed_pixel {
    BYTE r;
    BYTE g;
    BYTE b;
} RGB_PACKED_PIXEL;

typedef struct rgb_packed_image {
    int cols;
    int rows;
    RGB_PACKED_PIXEL **p;
    RGB_PACKED_PIXEL *data_p;
} RGB_PACKED_IMAGE;

typedef struct
{
    float *data;
    int width;
    int height;
}Image3;

typedef struct Image2 
{
	int width;
	int height;
	int channel;
	unsigned char* data;
}Image2;

typedef struct
{
	int width;
	int height;
	int channels; //ͼ��ͨ���� 
	unsigned char* Data;
}Image1;

typedef struct imageContainer {
    int x,y,n;
    unsigned char *data;
}image;

typedef struct _PGMData 
{
    int row;
    int col;
    int max_gray;
    int **matrix;
}PGMData;

struct hough_param_circle {
	int a;
	int b;
	int radius;
	int resolution;
	int thresh;
	struct point *points;
	int points_size;
};

typedef struct tagBGRA
{
	unsigned char blue;          // ����ɫ����ɫ����  (ֵ��ΧΪ0-255)
	unsigned char green;         // ����ɫ����ɫ����  (ֵ��ΧΪ0-255)
	unsigned char red;           // ����ɫ�ĺ�ɫ����  (ֵ��ΧΪ0-255)
	unsigned char transparency;  // ͸���ȣ���bmp���Ǳ���ֵ����ʵ��Ч��
}BGRA,*PBGRA;

typedef struct tagIMAGE
{
	unsigned int w;
	unsigned int h;
    BGRA* color;
}IMAGE,*PIMAGE;

typedef struct tagIGIMAGE
{
    unsigned int w;
    unsigned int h;
    int* date;
}IGIMAGE,*PIGIMAGE;

typedef struct {
    unsigned char B; //24λ��32λBMPͼ�����ɫͨ������ 
    unsigned char G; //24λ��32λBMPͼ�����ɫͨ������ 
    unsigned char R; //24λ��32λBMPͼ��ĺ�ɫͨ������ 
    unsigned char A; //����32λBMPͼ���Alphaͨ��
}BMPMat;

enum DebayerAlgorithm {
	NEARESTNEIGHBOUR,
	LINEAR
};

typedef enum type_morphing {
	SHRINKING = 0x1,
	THINNING  = 0x2,
	SKELETONIZING = 0x3
} type_morphing;

typedef struct {
    short B;
    short G; 
    short R; 
    short A;
}BMPMatshort;

typedef struct {
    int B;
    int G; 
    int R; 
    int A;
}BMPMatint;
 
typedef struct {
    float B;
    float G; 
    float R; 
    float A;
}BMPMatfloat;

typedef struct {
    double B;
    double G; 
    double R; 
    double A;
}BMPMatdouble;

typedef struct {
    char B;
    char G; 
    char R; 
    char A;
}BMPMatchar;

typedef unsigned short  uint16_t;

void HistogramEqualization5(char* input,char* output);
void Resize(char* input,char* output,int Height,int Width);
double MeanBrightness(char* input);
int IsBitMap(FILE *fp);
int getWidth(FILE *fp);
int getHeight(FILE *fp);
unsigned short getBit(FILE *fp);
unsigned int getOffSet(FILE *fp);
void Canny(char* input,char* output,int lowThreshold,int highThreshold);
void DCMtoBMP(string input,char* output);
void DES_Encrypt(char *PlainFile, char *Key,char *CipherFile);
void DES_Decrypt(char *CipherFile, char *Key,char *PlainFile);
void Ins1977(char* input,char* output,int ratio);
void KMeans1(char* input,char* output,int c,int k);
void KMeans(string input,unsigned int Clusters,char* output);
void LOMO(char* input,char* DarkAngleInput,char* output,int ratio);
void LSBRead(char* input,char* output,int width,int height,unsigned char color1,unsigned char color2);
void LSBWrite(char* input1,char* input2,char* output,int width,int height,unsigned char threshold1,unsigned char threshold2,unsigned char threshold3,unsigned char color1,unsigned char color2);
void MBVQ(char* input,char* output,int width,int height);
void OTSUBinarization(char* input,char* output);
void SegmentsOTSUBinarization(char* input,char* output);
void P3PPMBlur(char* input,char* output);
unsigned char** ReadPBM(char* input);
void WritePBM(unsigned char** Input,char* output);
void PGMSobel(char* input,char* output,int Mx[3][3],int My[3][3],int max,int min);
void PGMSobelX(char* input,char* output,int Mx[3][3],int My[3][3],int max,int min);
void PGMSobelY(char* input,char* output,int Mx[3][3],int My[3][3],int max,int min);
void PGMSobel1(char* input,char* output,int min,int max,int mx[3][3],int my[3][3]);
void PGMSobelX1(char* input,char* output,int min,int max,int mx[3][3],int my[3][3]);
void PGMSobelY1(char* input,char* output,int min,int max,int mx[3][3],int my[3][3]);
void PGMSobel2(char* input,char* XOutput,char* YOutput,char* SobelOutput,int sobel_x[3][3],int sobel_y[3][3],int min,int max);
void Sobel(char* input,char* output);
void Laplatian(char* input,char* output);
void HorizSobel(char* input,char* output);
void VertSobel(char* input,char* output);
void PGMSobel1(char* input,char* output,int threshold);
void PGMHistogramEqualization(char* input,char* output);
void PNGGray(char* input,char* output);
void PNGSpotlight(char* input,char* output,int centerX,int centerY,double a,double b,double c,double d,double e);
void PNGIllinify(char* input,char* output);
void PNGWaterMark(char* input1,char* input2,char* output);
PPMImage* ReadPPM(char* input);
void WritePPM(char* output,PPMImage* img);
void InvertColor(char* input,char* output);
void GrayFilter(char* input,char* output);
void SepiaFilter(char* input,char* output);
void AdjustSaturation(char* input,char* output,double a);
void Resize(char* input,char* output,unsigned int NewWidth, unsigned int NewHeight);
void AdjustHue(char* input,char* output,int a);
void AdjustBrightness(char* input,char* output,double a);
void AdjustContrast(char* input,char* output,double a);
void AdjustBlur(char* input,char* output,double a);
void MeanGrayFilter(char* input,char* output,double a);
void Pixelate(char* input,char* output,unsigned int a);
void Rotate(char* input,char* output,short a);
void GammaCorrection(char* input,char* output,double a);
void GrayAndChannelSeparation(char* input,char* Grayoutput,char* Routput,char* Goutput,char* Boutput);
void PGMBin(char* input,char* output,int threshold);
void Brightening(char* input,char* output,int a);
void GrayBrightening(char* input,char* output,int a);
void PPMFilter(char* input,char* output);
void PGMGrayFilter(char* input,char* output);
void PPMtoBMP(char* input,char* output);
void RAWtoPPM_red(char* input,char* output,int width, int height,DebayerAlgorithm algo);
void RAWtoPPM_green1(char* input,char* output,int width, int height,DebayerAlgorithm algo);
void RAWtoPPM_green2(char* input,char* output,int width, int height,DebayerAlgorithm algo);
void RAWtoPPM_blue(char* input,char* output,int width, int height,DebayerAlgorithm algo);
void RAWtoPPM(char* input,char* output,int width, int height,DebayerAlgorithm algo);
void RAWSobelEdge(char* input,char* output,int ROWS,int COLS,int M,float sobelX[3][3],float sobelY[3][3]);
void RAWPlaceHolder(char* input,char* output,int ROWS,int COLS,int M,float mask[3][3]);
void RAWLaplacialSharpeningFilter(char* input,char* output,int ROWS,int COLS,int M,float w,float mask[3][3]);
void RawLaplacianEnhancement(char* input1,char* output1,int width,int height);
void RawPowerTransformation(char* input,char* output,int width,int height,int c,float v);
void RAWAvgFilter(char* input,char* output,int ROWS,int COLS,int M,float mask[3][3]);
void RawImageInversion(char* input,char* output,int width,int height);
void RawHistogramEqualization(char* input,char* output,int width,int height);
void RAWHistogramEqualization(char* input,char* output,int width,int height);
void RAWMedianFilter(char* input,char* output,int ROWS,int COLS,int M,int sequence[9]);
void RawtoBmp1(char* input, char* output,unsigned long Width, unsigned long Height);
void RawToBmp(char* input,char* output,int imageWidth,int imageHigth);
void RGBtoCMY(string input,string output1,string output2,int height,int width,int NumberChannels,int a);
void RGBtoHSI(char* input,char* output);
void Roberts(unsigned char** input,unsigned char** output);
void Roberts(BMPMat** input,BMPMat** output);
void STLSection(char* input,char* output,int sliceAmount, int resolution,int c);
void SURF(char* input1,char* input2,char* output);
void SobelBinary(char* input,char* output);
void SobelOperator(char* input,char* output);
void Sobel2(char* input,char* output);
void Dark(char* input,char* output,int ratio);
void ClosedOperation(char* input,char* output);
void EdgeDetection(char* input,char* output);
void EdgeDetection1(char* input,char* output,short sharpen[3][3]);
void AdjustPixel(char* input,char* output,int a);
void EdgeDetection2(char* input,char* output,int a);
void EdgeDetection3(char* input,char* output,int a);
void EdgeDetection4(char* input,char* output,int a);
void YFiltering(char* input,char* output,int sobel_x[3][3],int sobel_y[3][3]);
void XFiltering(char* input,char* output,int sobel_x[3][3],int sobel_y[3][3]);
void SobelFiltering(char* input,char* output,int sobel_x[3][3],int sobel_y[3][3]);
void PrewittFiltering(char* input,char* output,int prewitt_x[3][3],int prewitt_y[3][3]);
void LaplacianFiltering(char* input,char* output,int laplacian[3][3]);
void SobelOperation1(char* input,char* output,int width,int height);
void SobelOperation2(char* input,char* output,int width,int height);
void Roberts(char* input,char* output);
void Prewitt(char* input,char* output);
void Sobel(char* input,char* output);
void Laplace(char* input,char* output);
void CyanGray(char* input,char* output,int width,int height);
void MagentaGray(char* input,char* output,int width,int height);
void YellowGray(char* input,char* output,int width,int height);
void PartialColorRetention(char* input,char* output,int ratio);
void GrayImageConversion8(char* input,char* output);
void Gray(char* input,char* output);
void GrayImageConversion(char* input,char* output);
void GrayLightness(string input,string output,int height,int width,int NumberChannels);
void GrayAverage(string input,string output,int height,int width,int NumberChannels);
void GrayLuminosity(string input,string output,int height,int width,int NumberChannels);
void Transfer(char* input,char* output,int width,int height);
void BinaryImageVerticalMirror(unsigned char *input,unsigned char *output,unsigned int w,unsigned int h);
void GrayImageVerticalMirror(unsigned char *input,unsigned char *output,unsigned int w,unsigned int h);
void ColorImageVerticalMirror(unsigned char *input,unsigned char *output,unsigned int w,unsigned int h);
void OTSU(char* input,char* output,int BeforeThreshold);
void PGMOtsuThreshold(string input,char* output);
void Homography(char* input1,char* input2,char* input3,char* output,int width,int height,int newwidth,int newheight);
void MovieEffect(char* input,char* output,int width,int height);
void LowerBrightness(char* input,char* output,int a,int b);
void HightBrightness(char* input,char* output,int a,int b);
void IterativeThresholdSelection(char* input,char* output);
void Dither(string input,string output,int height,int width,int NumberChannels,int method,int bayerMatrixNumber,int numberOfTones);
void AssimilateChannels(string input,string output,int height,int width,int NumberChannels,int method,int bayerMatrixNumber,int numberOfTones);
void DitheringMethod(char* input,char* output);
void FixedThresholdMethod(char* input,char* output,int width,int height);
void RandomThresholdMethod(char* input,char* output,int width,int height);
void DitherMatrixMethod(char* input,char* output,int width,int height,int N);
void LogTransformation(char* input,char* output,int constant);
void NormalizedLogBuffer1(char* input,char* output,int width,int height);
void NormalizedLogBuffer2(char* input,char* output,int width,int height);
void TernaryGrayLevel1(char* input,char* output,int width,int height);
void TernaryGrayLevel2(char* input,char* output,int width,int height);
void BestEdgeMap1(char* input,char* output,int width,int height);
void BestEdgeMap2(char* input,char* output,int width,int height);
void LogarithmicTransformation(char* input,char* output);
void HistogramEqualization(char* input,char* output);
void QRCodeGeneration(char *filename, char* inputString);
void Binarization(char* input,char* output,int threshold);
void Expansion(char* input,char* output,unsigned char mask[9],int c);
void Corrosion(char* input,char* output,unsigned char mask[9],int c);
void OpenOperation(char* input,char* output,unsigned char mask[9],int c);
void ClosedOperation(char* input,char* output,unsigned char mask[9],int c);
void OpenOperationToExtractContour(char* input,char* output,unsigned char mask[9],int c);
void ExpansionOperationToContourExtraction(char* input,char* output,unsigned char mask[9],int c);
void CorrosionCalculationToContourExtraction(char* input,char* output,unsigned char mask[9],int c);
void Glaw(char* input,char* output,int ratio);
void LowPassFilter(char* input,char* output);
void HighPassFilter(char* input,char* output);
void Thinning(char* input,char* output);
void ThinningLine(char* input,char* output);
void Corrosion(char* input,char* output);
void Corrosion1(char* input,char* output,int *TempBuf, int TempH, int TempW);
void Expand(char* input,char* output,int *TempBuf, int TempH, int TempW);
unsigned char** create2DImg(unsigned char* input, int w, int h);
unsigned char getMaxPixelWhole(unsigned char **input,int x,int y,int w,int h,int *Kernal,int kernalW,int halfKernalW);
unsigned char getMaxPixelCenter(unsigned char **input,int x,int y,int *Kernal,int kernalW,int halfKernalW);
unsigned char** imgDilate(unsigned char *input,int w,int h,int *Kernal,int kernalW,int halfKernalW);
unsigned char getMinPixelWhole(unsigned char **input,int x,int y,int w,int h,int *Kernal,int kernalW,int halfKernalW);
unsigned char getMinPixelCenter(unsigned char **input,int x,int y,int *Kernal,int kernalW,int halfKernalW);
unsigned char** imgErode(unsigned char *input,int w,int h,int *Kernal,int kernalW,int halfKernalW);
void Corrosion(unsigned char *input,unsigned char *output,int rows,int cols,int mat[5][5]);
void Expansion(unsigned char *input,unsigned char *output,int rows,int cols,int mat[5][5]);
void BoxBlurAdvanced(string input,string output,int radius);
void GaussianBlurFilter(char* input,char* output);
void GaussianFiltering(char* input,char* output);
void LaplaceEnhancement(char* input,char* output);
void Residual(char* input,char* output);
void Skeletonize(char* input,char* output,int width,int height);
void SunlightFilter(char* input,char* output,int intensity,int radius,int x,int y);
void Compress(char* input,char* output);
void Decompression(char* input, char* output);
void BlackWhite(char* input,char* output,int width,int height,unsigned char threshold1,unsigned char threshold2,unsigned char threshold3,unsigned char color1,unsigned char color2);
void BlackWhite(char* input,char* output);
void Underexposure(char* input,char* output);
void Overexposure(char* input,char* output);
void Nostalgia(char* input,char* Mask,char* output,int ratio);
void GammaTransform(char* input,char* output);
void GrayScale(char* input,char* output);
void GrayImageBinarization(char* input,char* output,int bit,int threshold);
void GreyPesudoColor(char* input,char* output);
void HoughTransform(char* input,char* output,unsigned char threshold);
static void EdgeDetectionWithoutNonmaximum(const LPCTSTR input, const LPCTSTR output,double a,double b,double c);
static void NonmaximumWithoutDoubleThresholding(const LPCTSTR input, const LPCTSTR output,double a,double b,double c);
static void CannyEdgeDetection(const LPCTSTR input, const LPCTSTR output,double a,double b,double c,int orank,int oranb);
static void HoughTransform(const LPCTSTR input, const LPCTSTR output,double a,double b,double c,int orank,int oranb);
void GrayLuminosity(string input,string output,int height,int width,int NumberChannels);
void DifferentiateImageSobelFilter(string input,string output,int height,int width,int NumberChannels);
void DifferentiateImageSobelFilterAndRGBtoCMY(string input,string output,int height,int width,int NumberChannels);
void EdgeDetectionSobelFilter(string input,string output,int height,int width,int NumberChannels,int threshold);
void RemoveSpeckles(string input,string output,int height,int width,int NumberChannels,int threshold,int background,int numberOfIterations,type_morphing morphingOperation);
void DoubleDifferentiatingGetEdges(string input,string output,int height,int width,int NumberChannels);
void GetTernaryMap(string input,string output,int height,int width,int NumberChannels,int threshold,string output_Histogram,bool writeHistogramToFile);
void BoxBlurBasic(string input,string output);
void CalculateCumulativeHistogramMap(char* input,char* outfile);
void Translation(string input,char* output,int dx,int dy);
void Mirrored(string input,char* output,char axis);
void Sheared(string input,char* output,char axis,double Coef);
void Scaled(string input,char* output,double cx,double cy);
void Rotated1(string input,char* output,double angle);
void SaltNoise(char* input,char* output,int a,int b,int c,int d);
void CrossProcess(char* input,char* output,int ratio);
vector<float> HarrisCornerDetection(char* input,int width,int height,int channels,int step,float threshold,float k,float sigma);
void PGMRotated(char* input,char* output,int width,int height,int channels,double theta);
void XCorner(char* input,char* output,int width,int height,int channels,double theta);
void YCorner(char* input,char* output,int width,int height,int channels,double theta);
void Smooth(char* input,char* output,int width,int height,int channels,float sigma_x,float sigma_y,double theta);
vector<float> HarrisCorner(char* input,char* output,int width,int height,int channels,float threshold,float k,float sigma);
void PGMLocalisedOtsuThreshold(string input,char* output);
void Conversion8(unsigned char** input,short** output);
void Conversion8(short** input,unsigned char** output);
void Conversion8(unsigned char** input,int** output);
void Conversion8(int** input,unsigned char** output);
void Conversion8(unsigned char** input,unsigned int** output);
void Conversion8(unsigned int** input,unsigned char** output);
void Conversion8(unsigned char** input,float** output);
void Conversion8(float** input,unsigned char** output);
void Conversion8(unsigned char** input,double** output);
void Conversion8(double** input,unsigned char** output);
void Conversion8(unsigned char** input,char** output);
void Conversion8(char** input,unsigned char** output);
void Conversion24(BMPMat** input,BMPMatshort** output);
void Conversion24(BMPMatshort** input,BMPMat** output);
void Conversion24(BMPMat** input,BMPMatint** output);
void Conversion24(BMPMatint** input,BMPMat** output);
void Conversion24(BMPMat** input,BMPMatfloat** output);
void Conversion24(BMPMatfloat** input,BMPMat** output);
void Conversion24(BMPMat** input,BMPMatdouble** output);
void Conversion24(BMPMatdouble** input,BMPMat** output);
void Conversion24(BMPMat** input,BMPMatchar** output);
void Conversion24(BMPMatchar** input,BMPMat** output);
void Conversion32(BMPMat** input,BMPMatshort** output);
void Conversion32(BMPMatshort** input,BMPMat** output);
void Conversion32(BMPMat** input,BMPMatint** output);
void Conversion32(BMPMatint** input,BMPMat** output);
void Conversion32(BMPMat** input,BMPMatfloat** output);
void Conversion32(BMPMatfloat** input,BMPMat** output);
void Conversion32(BMPMat** input,BMPMatdouble** output);
void Conversion32(BMPMatdouble** input,BMPMat** output);
void Conversion32(BMPMat** input,BMPMatchar** output);
void Conversion32(BMPMatchar** input,BMPMat** output);
void MeanFiltering(char* input,char* output);
void MeanFltering1(char* input,char* output);
void KapoorAlgorithm(char* input,char* output,int BeforeThreshold);
void OpenOperation(char* input,char* output);
void SeparableDiffusion(char* input,char* output,int width,int height);
void Diffusion(char* input,char* output,int ratio);
void LapulasFiltering(char* readPath,char* writePath,float CoefArray[9],float coef);
void ImageFiltering(char* input,char* output,float kernel[3][3]);
void ComicStrip(char* input,char* output,int ratio);
void BrightnessAdjustment1(char* input,char* output,int brightness,int contrast);
void BrightnessAdjustment2(char* input,char* output,int brightness,int contrast);
void ZeroFillingSymmetricExtension(char* input,char* output);
void PopArtStyle(char* input,char* output,int ratio);
void LightLeakage(char* input,char* Mask,char* output,int ratio);
void LinearFiltering(char* input,char* output,short average[3][3]);
void MedianFiltering(char* input,char* output,short average[3][3]);
void SharpeningFiltering(char* input,char* output,short average[3][3],short sharpen[3][3]);
void GradientSharpening(char* input,char* output,short average[3][3],short soble1[3][3],short soble2[3][3]);
void ArithmeticMeanFilter(char* input,char* output);
void GeometricMeanFilter(char* input,char* output);
void HarmonicMeanFilter(char* input,char* output);
void ContraHarmonicMeanFilter(char* input,char* output);
void Filter(char* input,char* output);
void GreenAtmosphere(char* input,char* output);
void GreenAtmosphere(char* input,char* output,int width,int height,double a);
void Mosaic(char* input,char* output,int x);
void MosaicFilter(char* input,char* output,int ratio);
int* TemplateMatching(char* input1,char* input2,char* output,unsigned char red,unsigned char green,unsigned char blue,double MatchScore);
double* TemplateMatching1(Image2* input, Image2* Template,char* output,char* output_txt,double threshold,int isWriteImageResult,unsigned char color,unsigned char red,unsigned char green,unsigned char blue);
double* TemplateMatching2(Image2* input, Image2* Template,char* output,char* output_txt,double threshold,int isWriteImageResult,unsigned char color,unsigned char red,unsigned char green,unsigned char blue);
Image2* TemplateMatching3(Image2* input, Image2* Template, char* output_txt, double threshold, int isWriteImageResult, unsigned char color, unsigned char red, unsigned char green, unsigned char blue);
Image2* TemplateMatching4(Image2* input, Image2* Template, char* output_txt, double threshold, int isWriteImageResult, unsigned char color, unsigned char red, unsigned char green, unsigned char blue);
double* TemplateMatching(RGB_PACKED_IMAGE* input, RGB_PACKED_IMAGE* Template,char* output,unsigned char red,unsigned char green,unsigned char blue,double c,double threshold);
RGB_PACKED_IMAGE* TemplateMatching(RGB_PACKED_IMAGE* input, RGB_PACKED_IMAGE* Template,unsigned char red, unsigned char green, unsigned char blue, double c, double threshold);
void TemplateMatching(char* input,char* templatename,char* output,unsigned int MaximumMatchingQuantity,double MatchScore,float suprapunereMaxima,unsigned char red,unsigned char green,unsigned char blue);
int* TemplateMatching(char* input1,char* input2,char* output);
int* TemplateMatching(char* input1,char* input2,char* output,float min);
double* TemplateMatch(char* input,char* templatefile,char* output,int size,int best_loss,double a,double b,double c,double d,int e1,int e2);
int* TemplateMatch(char* input,char* Template,char* output,int channels,int ROTATION);
int* TemplateMatch(image input,image Template,char* output,int channels,int ROTATION);
int* TemplateMatch(char* input,char* Template,int channels,int ROTATION);
int* TemplateMatch(image input,image Template,int ROTATION);
int ObjectFind(char* input,char* Templatefile);
void MuddyAtmosphere(char* input,char* output);
void Expansion(char* input,char* output);
void SmoothSharpen(char* input,char* output,int Template[3][3],int coefficient);
void GaussSmoothSharpen(char* input,char* output,int Template[3][3],int coefficient);
void SobelSharpen(char* input,char* output,int Templatex[3][3],int Templatey[3][3],int coefficient1,int coefficient2);
void MidSmoothing(char* input,char* output);
void AvgSmoothing(char* input,char* output);
void Averaging(char* input1,char* input2,char* input3,char* output,int a);
void PlaneSlicing(char* input,char* output);
void Translation(char* input,char* output,int xoffset,int yoffset);
void Denoising(char* input1,char* input2,char* output,int width,int height);
void SharpeningSpatialFiltering8(char* input,char* output,int model[9]);
void PseudoGrayscale(char* input,char* output);
void TwoColors(char* input,char* output,int threshold,unsigned char color1,unsigned char color2);
void Luminosity(char* input,char* output,int width,int height);
void Average(char* input,char* output,int width,int height);
void MinMax(char* input,char* output,int width,int height);
void PNGImageGeneration(char* filename,const unsigned char img[],unsigned W,unsigned H,int x);
void Shrink(char* input,char* output,int width,int height);
void BilateralFiltering(string input,char* output,double ssd, double sdid);
void DoubleLayerErosion(char* input,char* output);
void BilinearTransformation(char* input,char* output,int width,int height,int newwidth,int newheight);
void BilinearInterpolation(string input,string output,int height,int width,int NumberChannels,int targetHeight,int targetWidth);
void BinaryImageHorizontalMirror(unsigned char *input,unsigned char *output,unsigned int w,unsigned int h);
void GrayImageHorizontalMirror(unsigned char *input,unsigned char *output,unsigned int w,unsigned int h);
void ColorImageHorizontalMirror(unsigned char *input,unsigned char *output,unsigned int w,unsigned int h);
void DitherMatrixMethod(char* input,char* output,int width,int height,int N);
void SketchFilter(char* input,char* output,int ratio);
void Zoom(char* input,char* output,float scaleX,float scaleY,int interpolation);
void PGMSauvolaThreshold(string input,char* output,double a,double b,double c);
unsigned int FeatureDetection(char* input,char* output,unsigned char red,unsigned char green,unsigned char blue);
vector<int> FeatureMatching(char* input1,char* input2,char* output,unsigned char red,unsigned char green,unsigned char blue);
unsigned int FeatureMatching1(char* input1,char* input2,char* output,unsigned char red,unsigned char green,unsigned char blue,int OptimizationSwitch);
vector<int> FeatureMatching2(char* input1,char* input2,char* output,unsigned char red,unsigned char green,unsigned char blue,int OptimizationSwitch);
unsigned int FeatureExtraction1(char* input,char* output,unsigned char red,unsigned char green,unsigned char blue,int OptimizationSwitch);
std::list<ezsift::SiftKeypoint> FeatureExtraction2(char* input,char* output,unsigned char red,unsigned char green,unsigned char blue,int OptimizationSwitch);
int Equal(char* input1,char* input2,double c);
int GreaterThan(char* input1,char* input2,double c);
int LessThan(char* input1,char* input2,double c);
double GMSD(char* input1, char* input2);
void AddGaussNoise(char* input,char* output);
void AddSaltPepperNoise(char* input,char* output);
void ChannelSeparation(char* input,char* Routput,char* Goutput,char* Boutput);
void PatternMethod(char* input,char* output,unsigned char Template[8][8]);
void LayerAlgorithm(char*input,char* inputMix,char* output,int alpha,int blendModel);
void BMP24LossyCompression(char* input,char* output);
void BMP24LossyDecompression(char* input,char* output);
void BMP24LosslessCompression(char* input,char* output);
void BMP24LosslessDecompression(char* input,char* output);
void ImageDiscoloration(char* input,char* output,double a,double b,double c);
unsigned char** HorizontalConcavity(unsigned char** input,int RANGE,int height,int width);
unsigned char** HorizontalConvexity(unsigned char** input,int RANGE,int height,int width);
unsigned char** TrapezoidalDeformation(unsigned char** input,int height,int width,double k);
unsigned char** TriangularDeformation(unsigned char** input,int height,int width,double k);
unsigned char** SDeformation(unsigned char** input,int height,int width,int RANGE);
void ImageCutting(char* input,char* output,int leftdownx,int leftdowny,int rightupx,int rightupy);
void ImageLayerAlgorithm(char* input,char* output);
void RGBtoGraywithoutLUT(char* input,char* output);
void RGBtoGraywithLUT(char* input,char* output);
void PiecewiseLinearTransform(char* input,char* output);
void PowerConvertion(char* input,char* output,double c,double g);
void LaplacianEnhancement(char* input,char* output,int N,int LaplMask[3][3]);
void Smooth(char* input,char* output);
void LaplaceSmooth(char* input,char* output,int N,int LaplMask[3][3]);
void Sobel1(char* input,char* output,int N,int SblMask1[3][3],int SblMask2[3][3]);
void SobelSmooth(char* input,char* output,int N,int SblMask1[3][3],int SblMask2[3][3]);
void Multiply(char* input,char* output,int N,int SblMask1[3][3],int SblMask2[3][3],int LaplMask[3][3]);
void Add(char* input,char* output,int N,int SblMask1[3][3],int SblMask2[3][3],int LaplMask[3][3]);
void PowerConvertion1(char* input,char* output,double c,double g,int N,int SblMask1[3][3],int SblMask2[3][3],int LaplMask[3][3]);
void BlackWhite(char* input,char* output);
void RandomOperation(char* input,char* output,unsigned char treshold1,unsigned char treshold2,unsigned char treshold3,unsigned char treshold4,unsigned char treshold5,unsigned char treshold6,unsigned char red,unsigned char green,unsigned char blue,int color1,int color2,int color3,int color4,int color5,int color6,int color7,int color8);
void SpecialEffects1(char* input,char* output,unsigned char red,unsigned char green,unsigned char blue);
void NostalgicFilter(BMPMat** input,BMPMat** output);
void SizeTransformation(short** input,short** output,short height,short width,short out_height,short out_width);
void ReverseColor(short** input,short** output,long height,long width,short GRAY_LEVELS);
void Logarithm(short** input,short** output,long height,long width,short c);
void Gamma(short** input,short** output,long height,long width,double c);
void HistogramEqualization(short** input, short** output, long height, long width,short GRAY_LEVELS);
void SmoothLinearFiltering(short** input, short** output,long height, long width,short average[3][3]);
void MedianFiltering(short** input, short** output, long height, long width);
void Laplace(short** input,short** output,long height,long width,short sharpen[3][3]);
void Sobel(short** input,short** output,long height,long width,short soble1[3][3],short soble2[3][3]);
void DFTRead(short** input, double** output,long height,long width);
void DFTImaginary(short** input,double** output,long height,long width);
void FreSpectrum(short **input, short **output,long height,long width);
void IDFT(double** re_array,double** im_array,short** output,long height,long width);
void AddGaussianNoise(short** input,short** output,long height,long width);
void AddSaltPepperNoise(short** input, short** output,long height,long width);
void MeanFilter(short** input,short** output,long height,long width);
void GeometricMeanFilter(short** input,short** output,long height,long width,double product);
void HarmonicMeanFiltering(short** input,short** output,long height,long width,double sum);
void InverseHarmonicMeanFiltering(short** input,short** output,long height,long width,int Q);
void Threshold(short** input,short** output,long height,long width,int delt_t,double T);
void OTSU(short** input,short** output,long height,long width,short GRAY_LEVELS);
void MatrixGlobalAddition24(BMPMat** input1,BMPMat** input2,BMPMat** output);
void MatrixGlobalSubtraction24(BMPMat** input1,BMPMat** input2,BMPMat** output);
void MatrixGlobalMultiplication24(BMPMat** input1,BMPMat** input2,BMPMat** output);
void MatrixGlobalDivision24(BMPMat** input1,BMPMat** input2,BMPMat** output);
void MatrixGlobalAddition32(BMPMat** input1,BMPMat** input2,BMPMat** output);
void MatrixGlobalSubtraction32(BMPMat** input1,BMPMat** input2,BMPMat** output);
void MatrixGlobalMultiplication32(BMPMat** input1,BMPMat** input2,BMPMat** output);
void MatrixGlobalDivision32(BMPMat** input1,BMPMat** input2,BMPMat** output);
void MatrixGlobalAddition8(unsigned char** input1,unsigned char** input2,unsigned char** output);
void MatrixGlobalSubtraction8(unsigned char** input1,unsigned char** input2,unsigned char** output);
void MatrixGlobalMultiplication8(unsigned char** input1,unsigned char** input2,unsigned char** output);
void MatrixGlobalDivision8(unsigned char** input1,unsigned char** input2,unsigned char** output);
void ColorRectangleLocalSegmentation(char* input,char* output,int x1,int y1,int x2,int y2,BMPMat color);
void GrayRectangleLocalSegmentation(char* input,char* output,int x1,int y1,int x2,int y2,unsigned char color);
void ColorDrawRectangle(char* input,char* output,int x1,int y1,int x2,int y2,BMPMat color);
void GrayDrawRectangle(char* input,char* output,int x1,int y1,int x2,int y2,unsigned char color);
void Relief(BMPMat** input,BMPMat** output,int value);
void Relief(unsigned char** input,unsigned char** output,int value);
void Sharpening(BMPMat** input,BMPMat** output,double degree);
void Sharpening(unsigned char** input,unsigned char** output,double degree);
void Soften(BMPMat** input,BMPMat** output,int value);
void Soften(unsigned char** input,unsigned char** output,int value);
void flipX(char* input,char* output);
void flipY(char* input,char* output);
void Crop(char* input,char* output,uint16_t start_x, uint16_t start_y, uint16_t new_height, uint16_t new_width);
void Resize(char* input,char* output,int new_width, int new_height);
void Scale(char* input,char* output,double ratio);
void GrayscaleAvg(char* input,char* output);
void grayscaleLum(char* input,char* output);
void ColorMask(char* input,char* output,float r,float g,float b);
void PixeLize(char* input,char* output,int strength);
void GaussianBlur(char* input,char* output,int strength);
void EdgeDetection(char* input,char* output,double cutoff);
void Sharpen(char* input,char* output);
void GrayAVS(char* input,char* output,float k,float b);
void HistogramEqualize24(char* input,char* output);
void MatrixTransformation(char* input,char* output);
void Binarization(char* input,char* output);
void ChannelSeparation_B(char* input,char* output);
void ChannelSeparation_G(char* input,char* output);
void ChannelSeparation_R(char* input,char* output);
void Inverse(char* input,char* output);
void HistogramEqualization8(char* input,char* output);
void Smooth(char* input,char* output);
void CannyEdge(char* input,char* output);
void EdgeEnhance(char* input,char* output);
void AvrFilter(char* input,char* output1,char* output2,int M,int N);
void GryOppositionSSE(char* input,char* output);
void MedianFilter(char* input,char* output,int M,int N);
void EdgeSharpeningGry(char* input,char* output);
void SJGryandRiceTest(char* input,char* output);
void TextTest(char* input,char* output);
void RedChannel(char* input,char* output);
void GreenChannel(char* input,char* output);
void BlueChannel(char* input,char* output);
void HistogramStatistics(char* input,char* output);
void HistogramEqualization1(char* input,char* output);
void ReflectionRay(char* input,char* output);
void MeanFiltering24(char* input,char* output);
void MedianFiltering24(char* input,char* output);
void ZoomOutAndZoomIn(char* input,char* output,double value);
void Translation24(char* input,char* output,int x,int y);
void Mirror24(char* input,char* output);
void Rotate24(char* input,char* output,double degree);
void GivenThresholdMethod(char* input,char* output,int threshold);
void IterativeThresholdMethod(char* input,char* output);
void OstuThresholdSegmentationMethod(char* input,char* output);
void Repudiation(char* input,char* output);
void Gray1(char* input,char* output);
void CorrectMethod(char* input,char* output);
void ChannelSeparation1(char* input,char* Routput,char* Goutput,char* Boutput);
void ReverseColor(char* input,char* output);
Image1* LoadImage1(char* input);
void SaveImage1(char* output,Image1* img);
unsigned char** BMPRead8(char* input);
void GenerateImage8(char* output,unsigned char** color);
BMPMat** BMPRead(char* input);
unsigned int BMPHeight(char* input);
unsigned int BMPWidth(char* input);
void GenerateImage(char* output,BMPMat** color,unsigned short type);
void ImageContrastExtension(char* input,char* output,double m,double g1,double g2,double a);
void Binaryzation(char* input,char* output,int threshold);
void GlobalBinarization(char* input,char* output);
void AdaptiveBinarization(char* input,char* output);
void ExpansionOperation(char* input,char* output);
void CorrosionOperation(char* input,char* output);
void Operation1(char* input,char* output);
void Closed1(char* input,char* output);
void Negative1(char* input,char* output);
void Negative(char* input,char* output);
void ImageSynthesis(char* input1,char* input2,char* output);
void BlackWhite(char* input,char* output,float T,int border);
IMAGE Image_bmp_load(char* filename);
void Image_bmp_save(char* filename,IMAGE im);
IMAGE TransformShapeNearest(IMAGE input, unsigned int newWidth, unsigned int newHeight);
IMAGE TransformShapeLinear(IMAGE input, unsigned int newWidth, unsigned int newHeight);
IMAGE TransformShapeWhirl(IMAGE input, float angle);
IMAGE TransformShapeUpturn(IMAGE input, int a);
void TransformColorGrayscale(IMAGE im, int GrayscaleMode);
void TransformColorBWDIY(IMAGE input, unsigned char Threshold);
void TransformColorBWOSTU(IMAGE input);
void TransformColorBWTRIANGLE(IMAGE input);
IMAGE TransformColorBWAdaptive(IMAGE input, int areaSize);
IMAGE TransformColorBWGrayscale(IMAGE input, int areaSize);
void TransformColorOpposite(IMAGE input);
IMAGE TransformColorHistogramPart(IMAGE input);
IMAGE TransformColorHistogramAll(IMAGE input);
IMAGE KernelsUseDIY(IMAGE input, double* kernels, int areaSize, double modulus);
IMAGE WavefilteringMedian(IMAGE input);
IMAGE WavefilteringGauss(IMAGE input);
IMAGE Wavefiltering_LowPass(IMAGE input, double* kernels);
IMAGE WavefilteringHighPass(IMAGE input, double* kernels);
IMAGE Wavefiltering_Average(IMAGE input);
IMAGE EdgeDetectionDifference(IMAGE input, double* kernels);
IMAGE KernelsUseEdgeSobel(IMAGE input, double* kernels1, double* kernels2);
IMAGE EdgeDetectionLaplace(IMAGE input, double* kernels);
IMAGE MorphologyErosion(IMAGE input, double* kernels);
IMAGE MorphologyDilation(IMAGE input, double* kernels);
IMAGE Pooling(IMAGE input, int lenght);
IGIMAGE IntegralImage(IMAGE input);
void FaceDetection(char* input,char* output);
void IntegralDiagram(unsigned int *input, unsigned int *output, int width, int height);
void ImageEncryption(char* inFileName,char* outFileName,char key);
void EncryptionDecryption(char* input,char* output,int Key,int a);
void Encryption(char* input,char* output,int Key);
void Decryption(char* input,char* output,int Key);
void ImageDecryption(char* inFileName,char* outFileName,char key);
void Decompression(string input,string output);
void HorizontalMirror(char* input,char* output);
void MirrorVertically(char* input,char* output);
void XMirroring(char* input,char* output);
void YMirroring(char* input,char* output);
void ImageConvolution(char* input,char* output,double** Kernel,int n,int m);
void SpatialMeanFiter(char* input,char* output,int radius);
void SpatialMedianFiter(char* input,char* output,int radius);
void SpatialMaxFiter(char* input,char* output,int radius);
void SpatialMinFiter(char* input,char* output,int radius);
void SpatialGaussFiter(char* input,char* output,int radius);
void SpatialStatisticalFiter(char* input,char* output,int radius,float T);
float* ImageMatching(char* TargetImage,char* Template0,char* Template1,char* Template2,char* Template3,char* Template4,char* Template5,char* Template6,char* Template7,char* Template8,char* Template9);
void Mosaic(char* input,char* output,int w,int h);
void FFTAmp(char* input,char* output,bool inv);
void FFTPhase(char* input,char* output,bool inv);
void STDFT1(char* input,char* output,bool inv);
void STDFT2(char* input,char* output,bool inv);
void SpectrumShaping(char* input,char* inputMsk,char* output);
void Translation(char* input,char* output,int x,int y,unsigned char color);
void Nesting(char* Biginput,char* Smallinput,char* output);
void CrossDenoising24(BMPMat** input,BMPMat** output,BMPMat threshold,BMPMat target);
void CrossDenoising8(unsigned char** input,unsigned char** output,unsigned char threshold,unsigned char target);
void ImageDecontamination(BMPMat** input,BMPMat** output,int x1,int y1,int x2,int y2);
void ImageDecontamination(unsigned char** input,unsigned char** output,int x1,int y1,int x2,int y2);
void Blend(char* input1,char* input2,char* output);
void Checker(char* input1,char* input2,char* output);
void Blend1(char* input1,char* input2,char* output);
void Checker1(char* input1,char* input2,char* output);
void ImageSharpening(char* input,char* output);
void SharpenLaplace(char* input,char* output,int ratio);
void SharpenUSM(char* input,char* output,int radius,int amount,int threshold);
void DrawRectangle(char* input,char* output,int x1,int y1,int x2,int y2,unsigned char red,unsigned char green,unsigned char blue);
void GenerateBmp(unsigned char* pData,int width,int height,char* filename);
void Jpg24ImageGeneration(char* filename,unsigned int width, unsigned int height, unsigned char* img);
void ImageScalingNearestNeighborInterpolation(char* input,char* output,float lx,float ly);
void ImageScalingBilinearInterpolation(char* input,char* output,float lx,float ly);
void BilinearInterpolationScaling(char* input,char* output,float ExpScalValue);
void NearestNeighborInterpolationScaling(char* input,char* output,float ExpScalValue);
void ZoomImg(unsigned char *input,unsigned char *output,int sw,int sh,int channels,int dw,int dh);
void ImageFeatures(char* input,char* kernel,char* output);
void CrossDenoising24(BMPMat** input,BMPMat** output,BMPMat target,BMPMatdouble weight);
void CrossDenoising8(unsigned char** input,unsigned char** output,unsigned char target,double weight);
void RotateRight90Degrees(char* input,char* output);
void RotateLeft90Degrees(char* input,char* output);
void ImageRotation(char* input,char* output,double angle);
void Rotation8(char* input,char* output,double Angle,int x1,int y1,int x2,int y2,unsigned char color);
void Rotation24(char* input,char* output,double Angle,int x1,int y1,int x2,int y2,unsigned char red,unsigned char green,unsigned char blue);
void Rotation(char* input,char* output,int angle,unsigned char color);
void Rotate(char* input,char* output,int angle);
void imgRotate90Gray(unsigned char *input,unsigned char *output,int sw,int sh,int *dw,int *dh);
void imgRotate90Color(unsigned char *input,unsigned char *output,int sw,int sh,int *dw,int *dh);
void imgRotate270Gray(unsigned char *input,unsigned char *output,int sw,int sh,int *dw,int *dh);
void imgRotate270Color(unsigned char *input,unsigned char *output,int sw,int sh,int *dw,int *dh);
void imgRotate180Gray(unsigned char *Img,int w,int h);
void imgRotate180Color(unsigned char *Img,int w,int h);
void imgRBExchange(unsigned char *Img,int w,int h);
void Compress8(string input,string output);
void FileWrite(char* BMP,char* TXT);
void FileWriteOut(char* BMP,char* TXT);
void NoiseUniform(char* input,char* output,double a,double b);
void NoiseGauss(char* input,char* output,float mean,float delta);
void NoiseRayleigh(char* input,char* output,float a,float b);
void NoiseExp(char* input,char* output,float a);
void NoiseImpulse(char* input,char* output,float a,float b);
int ObjectsInImages(string input,string output,int height,int width,int NumberChannels,int threshold,float starSize,int a);
void Dewarped1(char* input,char* output,int width,int height);
void Dewarped2(char* input,char* output,int width,int height);
void grayToColor(FILE* input,FILE* output);
void Encode(char* input,char* output);
void Decode(char* input,char* output);
void FileDecompression(char *input , char *output);
void FileCompress(char *input , char *output);
void TextureSegmentation1(char* input,char* output,int width,int height,int K);
void TextureSegmentation2(char* input,char* output,int width,int height,int K);
void TextureClassification(vector <string> filename,char* output,int width,int height,int K,int N,int a);
void ErrorDiffusion1(char* input,char* output,int width,int height);
void ErrorDiffusion2(char* input,char* output,int width,int height);
void ErrorDiffusion3(char* input,char* output,int width,int height);
void ErrorDiffusion(string input,string output,int height,int width,int NumberChannels,int method,int kernelSize,int numberOfTones,bool useFilter);
void Thin(char* input,char* output,int width,int height);
void ImageThinning(char* input,char* output,char** str,int n,int m1,int a,int b);
int MinimumValueOfImagePixels(char* filename);
int MaximumValueOfImagePixels(char* filename);
float AverageValueOfImagePixels(char* filename);
double StandardDeviationOfImagePixels(char* filename);
double EntropyOfImage(char* filename);
float* CountTheFrequencyOfPixels(char* filename);
void BinaryMorphologicalFilteringComplete(string input,string output,int height,int width,int NumberChannels,int threshold,int numberOfIterations,type_morphing morphingOperation);
void Rotate(char* input,char* output,int angle,int interpolation);
void HSV(char* input,char* output,int h,int s,int v);
void OilPainting(char* input,char* output,int width,int height,int N);
void OilPainting1(char* input,char* output,int width,int height,int N);
void OilpaintFilter(char* input,char* output,int radius,int smooth);
void OilPaintingEffect1(string input,string output,int height,int width,int NumberChannels,int colorBits,bool bitsOK,int kernelSize,bool kernelSizeOK);
void OilPaintingEffect2(string input,string output,int height,int width,int NumberChannels,int colorBits,bool bitsOK,int kernelSize,bool kernelSizeOK);
void OilPaintingEffect3(string input,string output,int height,int width,int NumberChannels,int colorBits,bool bitsOK,int kernelSize,bool kernelSizeOK);
vector<int> DefectLocation(PGMData* Template,PGMData* Sample,int floor,int size,int a,int b,int c,int d,int e,int f,int g,int h,int FULL,int EMPTY,bool report);
vector<int> DefectSize(PGMData* Template,PGMData* Sample,int floor,int size,int a,int b,int c,int d,int e,int f,int g,int h,int FULL,int EMPTY,bool report);
vector<int> GoodBadQuantity(PGMData* Template,PGMData* Sample,int floor,int size,int a,int b,int c,int d,int e,int f,int g,int h,int FULL,int EMPTY,bool report);
unsigned int* CircleDetection(char* input);
struct hough_param_circle* CircleDetection(char* input,int width,int height);
void ImageWarpEllipticalGrid(string input,string output,int height,int width,int NumberChannels);
void HaloFilter(char* input,char* output,int ratio);
void GrayHistogram(char* input,char* output,int hWidth,int hHeight);
void RedHistogram(char* input,char* output,int hWidth,int hHeight);
void GreenHistogram(char* input,char* output,int hWidth,int hHeight);
void BlueHistogram(char* input,char* output,int hWidth,int hHeight);
void HistogramEqualization2(char* input,char* output,int imgBit);
void HistogramEqualization3(char* input,char* output);
void HistogramEqualization4(char* input,char* output);
void HistogramEqualization(char* input,char* output,int hWidth,int hHeight);
void GrayHistogramEqualization(char* input,char* output,int hWidth,int hHeight);
void RedHistogramEqualization(char* input,char* output,int hWidth,int hHeight);
void GreenHistogramEqualization(char* input,char* output,int hWidth,int hHeight);
void BlueHistogramEqualization(char* input,char* output,int hWidth,int hHeight);
void GrayScaleStretch(char* input,char* output,int hWidth,int hHeight);
void GrayHistagramStretch(char* input,char* output,int hWidth,int hHeight);
void RedHistagramStretch(char* input,char* output,int hWidth,int hHeight);
void GreenHistagramStretch(char* input,char* output,int hWidth,int hHeight);
void BlueHistagramStretch(char* input,char* output,int hWidth,int hHeight);
void MedianFiltering1(char* input,char* output);
void MedianFiltering2(char* input,char* output);
double CharacterRecognition(char* TargetImage,char* TemplateFileGroup[]);
void PGMThreshold(string input,char* output,int thresh);
void ThresholdProcessing(char* input,char* output,int Threshold);
void OTSUProcessing(char* input,char* output);
void BMPtoYUV(char* input,char* output, char yuvmode);
void BMPtoYUV420I(char* input,char* output);
void BMPtoYUV420II(char* input,char* output);
void YUVsuperposition(char* input1,char* input2,char* output,int width,int height,unsigned char Y_BLACK,unsigned char U_BLACK,unsigned char V_BLACK);
void YUVsuperposition(char* input1,char* input2,char* output,int width,int height,unsigned char Y_BLACK,unsigned char U_BLACK,unsigned char V_BLACK);
void YUVsuperposition(char* input1,char* input2,char* output,int width,int height,unsigned char Y_BLACK,unsigned char U_BLACK,unsigned char V_BLACK);
void YUV444toYUV420(char* input,char* output,int height,int width);
void YUV444toYUV420(char* input,char* output,int height,int width,int frames);
void YUVsuperposition(char* input1,char* input2,char* output,int width,int height,unsigned char Y_BLACK,unsigned char U_BLACK,unsigned char V_BLACK);
void YUVEdgeProcessingY(char* input,char* output,int width,int height,double k);
void YUVEdgeProcessingU(char* input,char* output,int width,int height,double k);
void YUVEdgeProcessingV(char* input,char* output,int width,int height,double k);
void BMPLoadedIntoYUV(char* inputBMP,char* inputYUV,char* output,int YUVwidth,int YUVheight,int depth,bool mt);
void YUVEdgeProcessingHorizontalDirection(char* input,char* output,int width,int height,double k);
void YUVVieoEdgeProcessing(char* input,char* output,int width,int height,int frame,int max_frame);
void YUVScale(char* input,char* output,int inputWidth,int inputHeight,int outputWidth,int outputHeight);
void NoiseTreatment(char* input,char* output,int width,int height,int TWICEwidth,int TWICEheight);
void NoiseTreatment(char* input,char* output,int width,int height,int frame,int max_frame);
void MakeSphere(double V[3],double S[3], double r, double a, double m, int ROWS, int COLS, char* output);
void MakeSphere(double vector_v[3],double vector_s[3], double r, double a, double m, int ROWS,int COLS,char* output);
