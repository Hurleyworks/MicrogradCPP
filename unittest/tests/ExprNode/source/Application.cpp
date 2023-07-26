#include "Jahley.h"

const std::string APP_NAME = "ExprNode";

#ifdef CHECK
#undef CHECK
#endif

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <json/json.hpp>
using nlohmann::json;


TEST_CASE("Test ExprNode class") {
    // ValuePtr is a typedef for shared_ptr<ExprNode>
    ValuePtr a;
    ValuePtr b;

    SUBCASE("Test Addition") {
        a = ExprNode::Create(1.0);
        b = ExprNode::Create(2.0);

        auto c = *a + b;

        CHECK(c->get_val() == doctest::Approx(3.0));
    }

    SUBCASE("Test Subtraction") {
        a = ExprNode::Create(5.0);
        b = ExprNode::Create(3.0);

        auto c = *a - b;

        CHECK(c->get_val() == doctest::Approx(2.0));
    }

    SUBCASE("Test Multiplication") {
        a = ExprNode::Create(2.0);
        b = ExprNode::Create(3.0);

        auto c = *a * b;

        CHECK(c->get_val() == doctest::Approx(6.0));
    }

    SUBCASE("Test Division") {
        a = ExprNode::Create(8.0);
        b = ExprNode::Create(2.0);

        auto c = *a / b;

        CHECK(c->get_val() == doctest::Approx(4.0));
    }

    SUBCASE("Test Backward Propagation") {
        a = ExprNode::Create(2.0);
        b = ExprNode::Create(3.0);

        // c = a * b
        auto c = *a * b;
        c->backward();

        CHECK(a->get_grad() == doctest::Approx(3.0));
        CHECK(b->get_grad() == doctest::Approx(2.0));
    }
}

TEST_CASE("More complex backpropagation tests") {
    SUBCASE("Test multiplication and addition with more variables") {
        auto a = ExprNode::Create(2.0);
        auto b = ExprNode::Create(3.0);
        auto c = ExprNode::Create(4.0);
        auto d = ExprNode::Create(5.0);

        // f = (a * b) + (c * d)
        auto f = *(*a * b) + (*c * d);
        CHECK(f->get_val() == doctest::Approx(26.0));

        f->backward();
        CHECK(a->get_grad() == doctest::Approx(3.0));
        CHECK(b->get_grad() == doctest::Approx(2.0));
        CHECK(c->get_grad() == doctest::Approx(5.0));
        CHECK(d->get_grad() == doctest::Approx(4.0));
    }

    SUBCASE("Test division and subtraction with more variables") {
        auto a = ExprNode::Create(10.0);
        auto b = ExprNode::Create(5.0);
        auto c = ExprNode::Create(2.0);
        auto d = ExprNode::Create(1.0);

        // f = (a / b) - (c / d)
        auto f = *(*a / b) - (*c / d);
        CHECK(f->get_val() == doctest::Approx(0.0));

        f->backward();
        CHECK(a->get_grad() == doctest::Approx(0.2));
        CHECK(b->get_grad() == doctest::Approx(-0.4));
        CHECK(c->get_grad() == doctest::Approx(-1.0));
        CHECK(d->get_grad() == doctest::Approx(2.0));
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
