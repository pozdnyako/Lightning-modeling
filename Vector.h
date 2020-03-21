#pragma once

struct Vector {
	Vector() :x(0.0), y(0.0) {}
	Vector(double _x, double _y) :x(_x), y(_y) {}

	double x, y;
};

Vector operator+ (const Vector&, const Vector&);
Vector operator- (const Vector&, const Vector&);
Vector operator* (const Vector&, const double&);

double operator* (const Vector&, const Vector&);
double cross(const Vector&, const Vector&);

struct Vector3 {
	Vector3() :x(), y(), z() {}
	Vector3(double _x, double _y, double _z) :x(_x), y(_y), z(_z) {}

	double x, y, z;

	Vector3 operator+ (const Vector3&);
	Vector3 operator- (const Vector3&);
	Vector3 operator* (const double &);

	double operator* (const Vector3&);
	Vector3 cross(const Vector3&, const Vector3&);
};

struct Vector3i {
	Vector3i() :x(), y(), z() {}
	Vector3i(int _x, int _y, int _z) :x(_x), y(_y), z(_z) {}

	int x, y, z;

	Vector3i operator+ (const Vector3i&);
	Vector3i operator- (const Vector3i&);
	Vector3i operator* (const int &);

	int operator* (const Vector3i&);
	Vector3i cross(const Vector3i&, const Vector3i&);
};