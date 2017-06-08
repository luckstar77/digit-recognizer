//
//  ALRect.hpp
//  shintao-recognize
//
//  Created by develop on 2017/5/23.
//  Copyright © 2017年 develop. All rights reserved.
//

#ifndef ALRect_hpp
#define ALRect_hpp

#include <stdio.h>

class ALRect {
public:
    ALRect();
    ALRect(int ltx, int lty, int width, int height, int count);
    
    int _ltx;
    int _lty;
    int _rdx;
    int _rdy;
    int _cx;
    int _cy;
    int _width;
    int _height;
    int _count;
    
    void SetLtx(int ltx);
    void SetLty(int lty);
    void SetRdx(int rdx);
    void SetRdy(int rdy);
    void AddCount(int increment);
private:
    
};

#endif /* ALRect_hpp */
