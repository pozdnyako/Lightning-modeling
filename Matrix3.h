#ifndef MATRIX3_H_INCLUDED
#define MATRIX3_H_INCLUDED

class Matrix3 {
public:
    Matrix3();
    Matrix3(int, int, int);
    Matrix3(const Matrix3&);

    virtual ~Matrix3();

    void    set_size(int, int, int);

    double  get_num(int, int, int);
    void    set_num(int, int, int, double);
    void    set_data(double*);

    int     get_sizeX() { return sizeX; }
    int     get_sizeY() { return sizeY; }
    int     get_sizeZ() { return sizeZ; }
    double* get_data()  { return data; }

    void    print(int);
private:

    int x_p(int);
    int y_p(int);
    int z_p(int);
    int p_xyz(int, int, int);

    void allocate_mem();

    double *data;
    int sizeX, sizeY, sizeZ;
    bool is_mem_allocated;
};

#endif // MATRIX3_H_INCLUDED
