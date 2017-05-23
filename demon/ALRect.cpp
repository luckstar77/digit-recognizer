//
//  ALRect.cpp
//  shintao-recognize
//
//  Created by develop on 2017/5/23.
//  Copyright © 2017年 develop. All rights reserved.
//

#include "ALRect.hpp"

// 預設建構函式
ALRect::ALRect() {
    _ltx = 0;
    _lty = 0;
    _rdx = 0;
    _rdy = 0;
    _cx = 0;
    _cy = 0;
    _width = 0;
    _height = 0;
    _count = 0;
    
//    void SetLtx();
//    void SetLty();
//    void SetRdx();
//    void SetRdy();
//    void AddCount();
}

ALRect::ALRect(int ltx, int lty, int width, int height, int count = 0) {
    _ltx = ltx;
    _lty = lty;
    _width = width;
    _height = height;
    _rdx = ltx + _width;
    _rdy = lty + _height;
    _cx = _ltx + _width / 2;
    _cy = _lty + _height / 2;
    _count = count;
}

void ALRect::SetLtx(int ltx) {
    _ltx = ltx;
    _width = _rdx - _ltx + 1;
    _cx = _ltx + _width / 2;
}

void ALRect::SetLty(int lty) {
    _lty = lty;
    _height = _rdy - _lty + 1;
    _cy = _lty + _height / 2;
}

void ALRect::SetRdx(int rdx) {
    _rdx = rdx;
    _width = _rdx - _ltx + 1;
    _cx = _ltx + _width / 2;
}

void ALRect::SetRdy(int rdy) {
    _rdy = rdy;
    _height = _rdy - _lty + 1;
    _cy = _lty + _height / 2;
}

void ALRect::AddCount(int increment) {
    _count += increment;
}
