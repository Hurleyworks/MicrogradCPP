#include "Jahley.h"

const std::string APP_NAME = "MicrogradCPP";

// The mean squared error loss function
ValuePtr meanSquardError(const std::vector<ValuePtr>& target, const std::vector<ValuePtr>& prediction)
{
	ValuePtr mse = ExprNode::Create(0);
	for (int i = 0; i < target.size(); i++)
	{
		ValuePtr diff = *target[i] - prediction[i];
		mse = *mse + *diff * diff;
	}
	return *mse / target.size();
}

// The stochastic gradient descent update rule
void gradientDescent(std::vector<ValuePtr> params, double lr)
{
	for (auto& p : params)
	{
		double grad = p->get_grad();

		p->set_val(p->get_val() + (-lr * grad));
	}
}
class Application : public Jahley::App
{
public:
	Application() :
		Jahley::App()
	{
		try
		{
			// Create an MLP with 2 hidden layers, each with 4 neurons
			MLP mlp(3, { 4, 4, 1 });

			std::vector<std::vector<ValuePtr>> inputs = {
				{ExprNode::Create(2.0), ExprNode::Create(3.0), ExprNode::Create(-1.0)},
				{ExprNode::Create(3.0), ExprNode::Create(-1.0), ExprNode::Create(0.5)},
				{ExprNode::Create(0.5), ExprNode::Create(1.0), ExprNode::Create(1.0)},
				{ExprNode::Create(1.0), ExprNode::Create(-1.0), ExprNode::Create(-1.0)} };

			std::vector<std::vector<ValuePtr>> targets = {
				{ExprNode::Create(1.0)},
				{ExprNode::Create(2.0)},
				{ExprNode::Create(3.0)},
				{ExprNode::Create(4.0)} };


			// Train 
			int epochs = 300;
			for (int epoch = 0; epoch < epochs; ++epoch)
			{
				LOG(DBUG) << "---------------------Epoch: " << epoch;
				// for (size_t i = 0; i < inputs.size(); ++i)
				for (size_t i = 0; i < 4; ++i)
				{
					std::vector<ValuePtr> in = inputs[i];
					LOG(DBUG) << "Input " << i << " " << in[0]->get_val() << ", " << in[1]->get_val() << ", " << in[2]->get_val();

					// Forward propagation
					std::vector<ValuePtr> prediction = mlp(inputs[i]);

					for (const auto& p : prediction)
					{
						LOG(DBUG) << "Prediction " << p->get_val();
					}

					// Calculate loss
					ValuePtr loss = meanSquardError(targets[i], prediction);

					// Zero gradients
					mlp.zero_grad();

					// Backward propagation
					loss->backward();

					// Update weights
					gradientDescent(mlp.parameters(), 0.01);

					LOG(DBUG) << "Loss: " << loss->get_val();
					LOG(DBUG) << " ";
				}
			}
		}
		catch (std::exception& e)
		{
			std::cout << e.what() << std::endl;
		}
	}

	~Application()
	{
	}

	void onCrash() override
	{
	}

private:
};

Jahley::App* Jahley::CreateApplication()
{
	return new Application();
}
