#pragma once

#include <string>

class Exception {
public:
	virtual std::string what() = 0;
};

class MemEx : public Exception {
public:
	virtual std::string what() {
		return "memory exception";
	}
};

class WrongCallEx : public MemEx {
public:
	virtual std::string what() {
		return MemEx::what() + ", wrong call";
	}
};

class AllocEx : public MemEx {
public:
	virtual std::string what() {
		return MemEx::what() + ", bad alloc";
	}
};

class FreeEx : public MemEx {
public:
	virtual std::string what() {
		return MemEx::what() + ", bad free";
	}
};