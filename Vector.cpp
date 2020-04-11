#include "Vector.h"

const Vector operator+(const Vector &a, const Vector &b) {
	return Vector(a.x + b.x, a.y + b.y);
}

const Vector operator-(const Vector &a, const Vector &b) {
	return Vector(a.x - b.x, a.y - b.y);
}

double operator*(const Vector &a, const Vector &b) {
	return a.x * b.x + a.y * b.y;
}

const Vector operator*(const Vector &vec, const double &k) {
	return Vector(vec.x * k, vec.y * k);
}

double cross(const Vector &a, const Vector &b) {
	return a.x * b.y - a.y * b.x;
}
