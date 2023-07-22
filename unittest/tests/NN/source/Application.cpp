#include "Jahley.h"

const std::string APP_NAME = "NN";

#ifdef CHECK
#undef CHECK
#endif

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <json/json.hpp>
using nlohmann::json;
#
TEST_CASE ("Layer Class Test")
{
    // Construct a layer with 3 input neurons and 2 output neurons
    Layer layer (3, 2);

    // Test successful Layer creation
    CHECK (layer.parameters().size() != 0);

    // Assuming a specific input to Layer
    std::vector<ValuePtr> input = {ExprNode::Create (1.0), ExprNode::Create (2.0), ExprNode::Create (3.0)};

    // Forward pass
    auto output = layer (input);

    // Test output size
    CHECK (output.size() == 2);

    // Assume a specific loss
    ValuePtr loss = *((*output[0] - ExprNode::Create (2.0))->pow (2)) + (*output[1] - ExprNode::Create (1.0))->pow (2);

    // Backward pass
    loss->backward();

    // Test gradients calculation
    for (auto& param : layer.parameters())
    {
        CHECK (param->get_grad() != 0.0);
    }

    // Test zero_grad
    layer.zero_grad();
    for (auto& param : layer.parameters())
    {
        CHECK (param->get_grad() == doctest::Approx (0.0));
    }
}

TEST_CASE ("Neuron Class Test")
{
    // Construct a Neuron with 3 inputs
    Neuron neuron (3);

    // Test successful Neuron creation
    CHECK (neuron.parameters().size() == 4); // 3 weights + 1 bias

    // Assuming a specific input to Neuron
    std::vector<ValuePtr> input = {ExprNode::Create (1.0), ExprNode::Create (2.0), ExprNode::Create (3.0)};

    // Forward pass
    auto output = neuron (input);

    // Check if output is a valid pointer
    CHECK (output != nullptr);

    // Assume a specific loss
    ValuePtr loss = (*output - ExprNode::Create (2.0))->pow (2);

    // Backward pass
    loss->backward();

    // Test gradients calculation
    for (auto& param : neuron.parameters())
    {
        CHECK (param->get_grad() != 0.0);
    }

    // Test zero_grad
    neuron.zero_grad();
    for (auto& param : neuron.parameters())
    {
        CHECK (param->get_grad() == doctest::Approx (0.0));
    }
}

class Application : public Jahley::App
{
 public:
    Application() :
        Jahley::App()
    {
        doctest::Context().run();
    }

 private:
};

Jahley::App* Jahley::CreateApplication()
{
    return new Application();
}
