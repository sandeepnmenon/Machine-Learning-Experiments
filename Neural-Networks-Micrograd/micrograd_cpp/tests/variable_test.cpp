#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <micrograd_cpp/variable.cpp>
using namespace micrograd_cpp;

int main(int, char**) {
    std::cout << "Hello, world!\n";
    auto a = make_shared<Variable>(-4.0f);
    auto b = make_shared<Variable>(2.0f);
    auto c = a + b;

    std::cout<<*c<<std::endl;

    return 0;
}
