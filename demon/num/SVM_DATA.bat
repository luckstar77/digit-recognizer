@echo off
set c= D:\OCR\digital-recognize\demon\num
pushd %c%
del SVM_DATA.tmp 2>nul
for /f "tokens=*" %%i in ('dir/b/s *.txt') do type "%%i">>SVM_DATA.txt
ren SVM_DATA.tmp SVM_DATA.txt
popd