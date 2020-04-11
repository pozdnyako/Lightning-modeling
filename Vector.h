#pragma once
#include<cmath>
#include<iostream>

struct Vector;
struct Vector3;
struct Vector3i;

struct Vector {
	Vector() :x(0.0), y(0.0) {}
	Vector(double _x, double _y) :x(_x), y(_y) {}

	double x, y;
};

const Vector operator+ (const Vector&, const Vector&);
const Vector operator- (const Vector&, const Vector&);
const Vector operator* (const Vector&, double);

double operator* (const Vector&, const Vector&);
double cross(const Vector&, const Vector&);

struct Vector3 {
	Vector3() :x(), y(), z() {}
	Vector3(double _x, double _y, double _z) :x(_x), y(_y), z(_z) {}
	Vector3(const Vector3i &a);

	double x, y, z;

	const Vector3 operator+ (const Vector3&) const;
	const Vector3 operator- (const Vector3&) const;
	const Vector3 operator* (double) const;
	const Vector3 operator/ (double) const;

	double operator* (const Vector3&) const;
};
const Vector3 cross(const Vector3&, const Vector3&);
std::ostream& operator<<(std::ostream &, const Vector3&);


struct Vector3i {
	Vector3i() :x(), y(), z() {}
	Vector3i(int _x, int _y, int _z) :x(_x), y(_y), z(_z) {}
	Vector3i(Vector3 a) :x(a.x), y(a.y), z(a.z) {}

	int x, y, z;

	const Vector3i operator+ (const Vector3i&) const;
	const Vector3i operator- (const Vector3i&) const;
	const Vector3i operator* (int) const;

	int operator* (const Vector3i&) const;

	double getNorm() const { return sqrt(x*x + y*y + z*z);  }
};

const Vector3i cross(const Vector3i&, const Vector3i&);