#include <opencv2/highgui/highgui.hpp>
#include "ALDigitRecognize.hpp"
#include <fstream>
#include <iostream>
#include <dirent.h>
#ifdef WINDOWS
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif

using namespace cv;
using namespace std;

int getdir(string dir, vector<string> &files);

int main(int argc,char** argv)
{
#ifdef TEST
    int count = 0;
    int success = 0;
    char org_dir[1024];
    getcwd(org_dir, 1024);
    string file = org_dir + string("/success.txt");
    string testFile = org_dir + string("/fail.txt");
    
    vector<string> folders = vector<string>();
    getdir(argv[4], folders);
    
    fstream fp;
    fp.open(file, ios::out);
    if(!fp){
        cout<<"Fail to open file: "<<file<<endl;
    }
    
    fstream fp1;
    fp1.open(testFile, ios::out);
    if(!fp1){
        cout<<"Fail to open testFile: "<<testFile<<endl;
    }
    
    for(int i=0; i<folders.size(); i++){
        string path = argv[4] + string("/") + folders[i];
        int type = 0;
        
        
        int idx = folders[i].find('.');
        string typeStr = folders[i].substr(0,idx);
        if (typeStr.find("F") != -1) typeStr = typeStr.substr(1, 3);
        cout << "type : " << typeStr << endl;;
        istringstream is(typeStr);
        is>>type;
        
        vector<string> files = vector<string>();
        getdir(path, files);
        
        for(int i=0; i<files.size(); i++){
            string file = path + string("/") + files[i];
            cout << file << endl;
            if(files[i].find(".bmp") != -1) {
                Mat image = imread(file,CV_LOAD_IMAGE_COLOR);
                unsigned char *result;
                
                if(!image.data) continue;
                
                idx = files[i].find('.');
                string comparisonStr = files[i].substr(0,idx);
                if (comparisonStr.find("F") != -1) comparisonStr = comparisonStr.substr(0, comparisonStr.find("F"));
                
                count++;
                
                result = ALDigitRecognize(type, image.data, argv[2]);
                
                printf("callback : %d, %c, %c, %c, %c, %c\n", result[0], result[1], result[2], result[3], result[4], result[5]);
                
                bool isSuccess = !result[0];
                for(int k = 0; k < comparisonStr.length(); k++) {
                    if(result[k+1] != comparisonStr.at(k)) {
                        isSuccess = false;
                        break;
                    }
                 }
                
                if(isSuccess) {
                    fp << file << endl;
                    fp << result[1] << result[2] << result[3] << result[4] << result[5] << endl;
                    success++;
                } else {
                    fp1 << file << endl;
                    fp1 << result[1] << result[2] << result[3] << result[4] << result[5] << endl;
                }
            }
        }
    }
    
    cout << "Status" << endl;
    cout << "Count : " << count  << endl;
    cout << "Success : " << success  << endl;
    cout << "Fail : " << count - success << endl;
    cout << "Recognition Percent : " << (int)(success / (float)count * 100) << "%" << endl;
    
    fp << "Status" << endl;
    fp << "Count : " << count  << endl;
    fp << "Success : " << success  << endl;
    fp << "Fail : " << count - success << endl;
    fp << "Recognition Percent : " << (int)(success / (float)count * 100) << "%" << endl;
    
    fp.close();
    fp1.close();
#else
    Mat image = imread(argv[1],CV_LOAD_IMAGE_COLOR);
    unsigned char *result;
    if(!image.data)
    {
        return -1;
    }
    
    result = ALDigitRecognize(atoi(argv[3]), image.data, argv[2]);
    
    printf("callback : %d, %c, %c, %c, %c, %c", result[0], result[1], result[2], result[3], result[4], result[5]);
#endif
    
    //system("pause");
    waitKey(0);
}

int getdir(string dir, vector<string> &files){
    DIR *dp;//?µç?è³‡æ?å¤¾æ?æ¨?
    struct dirent *dirp;
    if((dp = opendir(dir.c_str())) == NULL){
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }
    while((dirp = readdir(dp)) != NULL){//å¦‚æ?dirent?‡æ??žç©º
        files.push_back(string(dirp->d_name));//å°‡è??™å¤¾?Œæ?æ¡ˆå??¾å…¥vector
    }
    closedir(dp);//?œé?è³‡æ?å¤¾æ?æ¨?
    return 0;
}
