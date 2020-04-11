#include "Parameters.h"

#include <cstdio>
#include <cstring>
#include <string>

bool checkString(const char *str1, int &i, int len1, const char *str2, void* num, const char* format) {
    int len2 = strlen(str2);

    
    while(strncmp(str1 + i, str2, len2) != 0) {
        if(len2 > len1 - i - 1) {
            return false;
        }
        i++;
    }

    int val_len = sscanf(str1 + i + len2, format, num);
    i += len2 + val_len;
}

Parameters::Parameters(std::string path) {
    FILE *file = fopen(path.c_str(), "rt");
       
    const int BUF_SIZE = 1000;
    char* buf = new char[BUF_SIZE];

    fread(buf, sizeof(char), BUF_SIZE, file);

    int len = strlen(buf);
    int i = 0;
    checkString(buf, i, len, "SIZE=",   &SIZE,   "%d");
    checkString(buf, i, len, "SIZE_Z=", &SIZE_Z, "%d");
    checkString(buf, i, len, "dL=",     &dL,     "%lf");
    checkString(buf, i, len, "U_0=",    &U_0,    "%lf");
    checkString(buf, i, len, "EPS=",    &EPS,    "%lf");
    checkString(buf, i, len, "SQ_SIZE=",&SQ_SIZE,"%d");

    fclose(file);

    if(SQ_SIZE > 0) {
        SCREEN_X = 3 * SQ_SIZE * SIZE;
        SCREEN_Y =     SQ_SIZE * SIZE;
        SECTION_SIZE = SQ_SIZE * SIZE;
    }
    else {
        SCREEN_X = 3 * SIZE / abs(SQ_SIZE);
        SCREEN_Y =     SIZE / abs(SQ_SIZE);
        SECTION_SIZE = SIZE / abs(SQ_SIZE);
    }
}

std::ostream& operator<<(std::ostream& out, const Parameters& param) {
    out << "Parameters:" << std::endl;
    out << "\tSIZE\t" << param.SIZE << std::endl;
    out << "\tSIZE_Z\t" << param.SIZE_Z << std::endl;
    out << "\tdL\t" << param.dL << " m" << std::endl;
    out << "\tU_0\t" << param.U_0 << " kv"  << std::endl;
    out << "\tEPS\t" << param.EPS << std::endl;
    out << "\tSQ_SIZE\t" << param.SQ_SIZE << std::endl;
    
    return out;
}
