#pragma once

#include "Vector.h"
#include "Exception.h"

template<class T>
bool is_in(T x, T a, T b) {
	return (x >= a && x <= b) || (x >= b && x <= a);
}

class MathEx : public Exception {
	virtual std::string what() {
		return "math exception";
	}
};

