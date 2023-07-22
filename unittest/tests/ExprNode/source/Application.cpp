#include "Jahley.h"

const std::string APP_NAME = "ExprNode";

#ifdef CHECK
#undef CHECK
#endif

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <json/json.hpp>
using nlohmann::json;


TEST_CASE ("Testing the ExprNode class with complex operations")
{
    /* Python versions from https://github.com/karpathy/micrograd
        a = ExprNode(-4.0)
        b = ExprNode(2.0)
        c = a + b
        d = a * b + b**3
        c += c + 1
        c += 1 + c + (-a)
        d += d * 2 + (b + a).relu()
        d += 3 * d + (b - a).relu()
        e = c - d
        f = e**2
        g = f / 2.0
        g += 10.0 / f

    */

    // a = ExprNode(-4.0)
    ValuePtr a = ExprNode::Create (-4.0);

    // b = ExprNode (2.0)
    ValuePtr b = ExprNode::Create (2.0);

    // c = a + b
    ValuePtr c = *a + b;
    CHECK (c->get_val() == -2.0);

    // d = a* b + b** 3
    ValuePtr d = *(*a * b) + (b->pow (3));
    CHECK (d->get_val() == 0.0);

    // c += 1 + c + (-a)
    c = *(*c + c) + 1;
    CHECK (c->get_val() == -3.0);

    // c += 1 + c + (-a)
    c = *(*c + 1) + (*c + (-(*a)));
    CHECK (c->get_val() == -1.0);

    // d += d * 2 + (b + a).relu()
    d = *(*d + (*d * 2)) + (*b + a)->relu();
    CHECK (d->get_val() == 0.0);

    // d += 3 * d + (b - a).relu()
    d = *(*d + *d * 3) + (*b - a)->relu();
    CHECK (d->get_val() == 6.0);

    // e = c - d
    ValuePtr e = *c - d;
    CHECK (e->get_val() == -7.0);

    // f = e**2
    ValuePtr f = e->pow (2);
    CHECK (f->get_val() == 49.0);

    // g = f / 2.0
    ValuePtr g = *f / 2.0;
    CHECK (g->get_val() == 24.5);

    // g += 10.0 / f
    g = *g + (*ExprNode::Create (10.0) / f);
    CHECK (g->get_val() == 24.70408163265306);

    g->backward();
    CHECK (a->get_grad() == doctest::Approx (138.8338));
    CHECK (b->get_grad() == doctest::Approx (645.5773));
}

TEST_CASE ("Testing the ExprNode class")
{
    // Test addition
    ValuePtr v1 = ExprNode::Create (5.0);
    ValuePtr v2 = ExprNode::Create (3.0);
    ValuePtr v3 = (*v1) + v2;
    ValuePtr v4 = (*v3) + 4;

    CHECK (v1->get_val() == 5.0);
    CHECK (v2->get_val() == 3.0);
    CHECK (v3->get_val() == 8.0);
    CHECK (v4->get_val() == 12.0);

    // Test multiplication
    v1 = ExprNode::Create (6.0);
    v2 = ExprNode::Create (2.0);
    v3 = (*v1) * v2;
    v4 = (*v3) * 100;

    CHECK (v1->get_val() == 6.0);
    CHECK (v2->get_val() == 2.0);
    CHECK (v3->get_val() == 12.0);
    CHECK (v4->get_val() == 1200.0);

    // Test power
    v1 = ExprNode::Create (4.0);
    v3 = v1->pow (2.0);

    CHECK (v1->get_val() == 4.0);
    CHECK (v3->get_val() == 16.0);

    // Test negation
    v1 = ExprNode::Create (-3.0);
    v2 = -(*v1);

    CHECK (v1->get_val() == -3.0);
    CHECK (v2->get_val() == 3.0);

    // Test backward
    v1 = ExprNode::Create (4.0);
    v2 = ExprNode::Create (2.0);
    v3 = (*v1) * v2;
    v3->backward();

    CHECK (v1->get_grad() == 2.0);
    CHECK (v2->get_grad() == 4.0);

    // Test ReLU
    v1 = ExprNode::Create (-1.0);
    v2 = v1->relu();
    CHECK (v2->get_val() == 0.0);

    v1 = ExprNode::Create (1.0);
    v2 = v1->relu();
    CHECK (v2->get_val() == 1.0);
}

TEST_CASE ("Testing ExprNode operations")
{
    ValuePtr a = ExprNode::Create (5.0);
    ValuePtr b = ExprNode::Create (3.0);

    SUBCASE ("Testing Addition")
    {
        ValuePtr c = *a + b;
        CHECK (c->get_val() == 8.0);
    }

    SUBCASE ("Testing Subtraction")
    {
        ValuePtr c = *a - b;
        CHECK (c->get_val() == 2.0);
    }

    SUBCASE ("Testing Multiplication")
    {
        ValuePtr c = *a * b;
        CHECK (c->get_val() == 15.0);
    }

    SUBCASE ("Testing Division")
    {
        ValuePtr c = *a / b;
        CHECK (c->get_val() == doctest::Approx (1.66667).epsilon (0.00001));
    }

    SUBCASE ("Testing Power")
    {
        ValuePtr c = a->pow (2);
        CHECK (c->get_val() == 25.0);
    }

    SUBCASE ("Testing ReLU Activation - Positive Input")
    {
        ValuePtr c = a->relu();
        CHECK (c->get_val() == 5.0);
    }

    SUBCASE ("Testing ReLU Activation - Negative Input")
    {
        ValuePtr a_neg = ExprNode::Create (-5.0);
        ValuePtr c = a_neg->relu();
        CHECK (c->get_val() == 0.0);
    }

    SUBCASE ("Testing Division By Zero")
    {
        ValuePtr zero = ExprNode::Create (0.0);
        CHECK_THROWS_AS (*a / zero, std::invalid_argument);
    }
}

TEST_CASE ("Testing ExprNode backpropagation")
{
    ValuePtr a = ExprNode::Create (5.0);
    ValuePtr b = ExprNode::Create (3.0);

    SUBCASE ("Testing backpropagation with addition")
    {
        ValuePtr c = *a + b;
        c->backward();
        CHECK (a->get_grad() == doctest::Approx (1.0).epsilon (0.00001));
        CHECK (b->get_grad() == doctest::Approx (1.0).epsilon (0.00001));
    }

    SUBCASE ("Testing backpropagation with subtraction")
    {
        ValuePtr c = *a - b;
        c->backward();
        CHECK (a->get_grad() == doctest::Approx (1.0).epsilon (0.00001));
        CHECK (b->get_grad() == doctest::Approx (-1.0).epsilon (0.00001));
    }

    SUBCASE ("Testing backpropagation with multiplication")
    {
        ValuePtr c = *a * b;
        c->backward();
        CHECK (a->get_grad() == doctest::Approx (3.0).epsilon (0.00001));
        CHECK (b->get_grad() == doctest::Approx (5.0).epsilon (0.00001));
    }

    SUBCASE ("Testing backpropagation with division")
    {
        ValuePtr c = *a / b;
        c->backward();
        CHECK (a->get_grad() == doctest::Approx (1.0 / 3.0).epsilon (0.00001));
        CHECK (b->get_grad() == doctest::Approx (-5.0 / 9.0).epsilon (0.00001));
    }

    SUBCASE ("Testing backpropagation with power")
    {
        ValuePtr c = a->pow (2);
        c->backward();
        CHECK (a->get_grad() == doctest::Approx (10.0).epsilon (0.00001));
    }

    SUBCASE ("Testing backpropagation with ReLU activation - Positive Input")
    {
        ValuePtr c = a->relu();
        c->backward();
        CHECK (a->get_grad() == doctest::Approx (1.0).epsilon (0.00001));
    }

    SUBCASE ("Testing backpropagation with ReLU activation - Negative Input")
    {
        ValuePtr a_neg = ExprNode::Create (-5.0);
        ValuePtr c = a_neg->relu();
        c->backward();
        CHECK (a_neg->get_grad() == doctest::Approx (0.0).epsilon (0.00001));
    }
}

TEST_CASE ("Testing ExprNode complex operations")
{
    ValuePtr a = ExprNode::Create (5.0);
    ValuePtr b = ExprNode::Create (3.0);
    ValuePtr c = ExprNode::Create (2.0);

    SUBCASE ("Testing complex operation 1: (a + b) * c")
    {
        ValuePtr res = *(*a + b) * c;
        res->backward();
        CHECK (a->get_grad() == doctest::Approx (2.0).epsilon (0.00001));
        CHECK (b->get_grad() == doctest::Approx (2.0).epsilon (0.00001));
        CHECK (c->get_grad() == doctest::Approx (8.0).epsilon (0.00001));
    }

    SUBCASE ("Testing complex operation 2: a / (b + c)")
    {
        ValuePtr res = *a / (*b + c);
        res->backward();
        CHECK (a->get_grad() == doctest::Approx (1.0 / 5.0).epsilon (0.00001));
        CHECK (b->get_grad() == doctest::Approx (-5.0 / 25.0).epsilon (0.00001)); // Adjusted this line
        CHECK (c->get_grad() == doctest::Approx (-5.0 / 25.0).epsilon (0.00001)); // Adjusted this line
    }

    SUBCASE ("Testing complex operation 3: relu(a) + b * c")
    {
        ValuePtr res = *(a->relu()) + (*b * c);
        res->backward();
        CHECK (a->get_grad() == doctest::Approx (1.0).epsilon (0.00001));
        CHECK (b->get_grad() == doctest::Approx (2.0).epsilon (0.00001));
        CHECK (c->get_grad() == doctest::Approx (3.0).epsilon (0.00001));
    }

    SUBCASE ("Testing complex operation 4: (a * b) / (c + 1)")
    {
        ValuePtr res = *(*a * b) / (*c + 1);
        res->backward();
        CHECK (a->get_grad() == doctest::Approx (3.0 / 3.0).epsilon (0.00001));
        CHECK (b->get_grad() == doctest::Approx (5.0 / 3.0).epsilon (0.00001));
        CHECK (c->get_grad() == doctest::Approx (-15.0 / 9.0).epsilon (0.00001));
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
