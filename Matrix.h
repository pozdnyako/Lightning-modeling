#ifndef MATRIX_H_INCLUDED
#define MATRIX_H_INCLUDED

class Matrix {
public:
    Matrix();
    Matrix(int, int);

    virtual ~Matrix();

    void    set_size(int, int);

    double  get_num(int, int);
    void    set_num(int, int, double);
    void    set_data(double*);

    int     get_sizeX() { return sizeX; }
    int     get_sizeY() { return sizeY; }

    void    print();
private:

    int x_p(int);
    int y_p(int);
    int p_xy(int, int);

    void allocate_mem();

    double *data;
    int sizeX, sizeY;
    bool is_mem_allocated;
};


#endif // MATRIX_H_INCLUDED
