#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <micrograd_cpp/variable.cpp>
using namespace std;

int main(int, char**) {
    std::cout << "Hello, world!\n";
    // Inputs x1 and x2
    auto x1 = make_shared<micrograd_cpp::Variable>(2.0f);
    x1->label("x1");
    auto x2 = make_shared<micrograd_cpp::Variable>(1.0f);
    x2->label("x2");

    // Two weights for each input
    auto w1 = make_shared<micrograd_cpp::Variable>(-3.0f);
    w1->label("w1");
    auto w2 = make_shared<micrograd_cpp::Variable>(0.0f);
    w2->label("w2");

    // Bias
    auto b = make_shared<micrograd_cpp::Variable>(8.0f);
    b->label("b");

    // Applying sum of inputs with weights
    auto x1w1 = x1 * w1;
    x1w1->label("x1w1");
    auto x2w2 = x2 * w2;
    x2w2->label("x2w2");
    auto x1w1x2w2 = x1w1 + x2w2;
    x1w1x2w2->label("x1w1x2w2");
    auto neuron1=  x1w1x2w2 + b;
    neuron1->label("neuron1");
    auto output = neuron1->tanh();
    output->label("output");

    cout<<*output<<endl;
    // Backpropagation
    output->backward();

    cout<<*output<<endl;
    cout<<*neuron1<<endl;

    cout<<*w1<<endl;
    cout<<*w2<<endl;
  
    return 0;
}
