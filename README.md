# I modified .. (by Hyun-Koo KIM)
## It is LWIR Camera based on Vehicle Detection Using DetectNet
* git clone
* mkdir build
* cd build
* cmake ../
* make

## Database download link is
* http://naver.me/FIIJ3Hfn
* Database download and uncompress
* if database is not "/home/ubuntu/GITC_LWIR"
* you are modify in /detectnet-lwir/detectnet-lwir.cpp
* in the 93 line, 
* modify filepath and filename of "sprintf(imgFilename, "/home/ubuntu/GITC_LWIR/DB_1st/images/%06d.bmp", num);"
* And, in the build folder, make
* ./detectnet-lwir

### Project Simulation Result
[![Results](https://img.youtube.com/vi/4u2bGj19x9k/0.jpg)](https://www.youtube.com/watch?v=4u2bGj19x9k)

