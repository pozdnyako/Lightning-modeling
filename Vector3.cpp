#include "Vector.h"

Vector3::Vector3(const Vector3i &a) :x(a.x), y(a.y), z(a.z) {}

std::ostream& operator<<(std::ostream & os, const Vector3& vec) {
	os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
	return os;
}

const Vector3 Vector3::operator+(const Vector3 &vec) const {
	return Vector3(x + vec.x,
				   y + vec.y,
				   z + vec.z);
}

const Vector3 Vector3::operator-(const Vector3 &vec) const {
	return Vector3(x - vec.x,
				   y - vec.y,
				   z - vec.z);
}

double Vector3::operator*(const Vector3 &vec) const {
	return x * vec.x + y * vec.y;
}

const Vector3 Vector3::operator*(double k) const {
	return Vector3(x * k,
				   y * k,
				   z * k);
}

const Vector3 Vector3::operator/(double k) const {
	return Vector3(x / k,
				   y / k,
				   z / k);
}


const Vector3 cross(const Vector3 &a, const Vector3 &b){
	return Vector3(a.y * b.z - b.y * a.z,
				   a.z * b.x - a.x * b.z,
				   a.x * b.y - a.y * b.x);
}

//	---------------------------------------------------------
//	---------------------  Vector3i  ------------------------
//	---------------------------------------------------------


const Vector3i Vector3i::operator+(const Vector3i &vec) const {
	return Vector3i(x + vec.x,
				    y + vec.y,
				    z + vec.z);
}

const Vector3i Vector3i::operator-(const Vector3i &vec) const {
	return Vector3i(x - vec.x,
				    y - vec.y,
				    z - vec.z);
}

int Vector3i::operator*(const Vector3i &vec) const {
	return x * vec.x + y * vec.y;
}

const Vector3i Vector3i::operator*(int k) const {
	return Vector3i(x * k,
				    y * k,
				    z * k);
}

const Vector3i cross(const Vector3i &a, const Vector3i &b) {
	return Vector3i(a.y * b.z - b.y * a.z,
				    a.z * b.x - a.x * b.z,
				    a.x * b.y - a.y * b.x);
}