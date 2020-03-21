#include "Vector.h"

Vector3 Vector3::operator+(const Vector3 &vec) {
	return Vector3(x + vec.x,
				   y + vec.y,
				   z + vec.z);
}

Vector3 Vector3::operator-(const Vector3 &vec) {
	return Vector3(x - vec.x,
				   y - vec.y,
				   z - vec.z);
}

double Vector3::operator*(const Vector3 &vec) {
	return x * vec.x + y * vec.y;
}

Vector3 Vector3::operator*(const double &k) {
	return Vector3(x * k,
				   y * k,
				   z * k);
}

Vector3 cross(const Vector3 &a, const Vector3 &b) {
	return Vector3(a.y * b.z - b.y * a.z,
				   a.z * b.x - a.x * b.z,
				   a.x * b.y - a.y * b.x);
}

//	---------------------------------------------------------
//	---------------------  Vector3i  ------------------------
//	---------------------------------------------------------


Vector3i Vector3i::operator+(const Vector3i &vec) {
	return Vector3i(x + vec.x,
				    y + vec.y,
				    z + vec.z);
}

Vector3i Vector3i::operator-(const Vector3i &vec) {
	return Vector3i(x - vec.x,
				    y - vec.y,
				    z - vec.z);
}

int Vector3i::operator*(const Vector3i &vec) {
	return x * vec.x + y * vec.y;
}

Vector3i Vector3i::operator*(const int &k) {
	return Vector3i(x * k,
				    y * k,
				    z * k);
}

Vector3i cross(const Vector3i &a, const Vector3i &b) {
	return Vector3i(a.y * b.z - b.y * a.z,
				    a.z * b.x - a.x * b.z,
				    a.x * b.y - a.y * b.x);
}