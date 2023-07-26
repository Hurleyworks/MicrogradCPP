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
void gradientDescent(const std::vector<ValuePtr>& params, double learningRate)
{
	for (auto& p : params)
	{
		double gradient = p->get_grad();

		p->set_val(p->get_val() - learningRate * gradient);
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

			// The tanh function outputs values in the range -1 to 1. 
			// If you're using tanh as the activation function in the output
			// layer of your MLP and you have targets outside this range, 
			// then it's not possible for the MLP to correctly predict those targets.
			std::vector<std::vector<ValuePtr>> targets = {
				{ExprNode::Create(-1.0)},
				{ExprNode::Create(1.0)},
				{ExprNode::Create(1.0)},
				{ExprNode::Create(-1)} };


			// "Epoch" is a term used in machine learning to denote one complete pass 
			// through the entire training dataset. It's used as a measure of the number
			// of times the learning algorithm has worked through the entire training set.
			int epochs = 1000;
			double learningRate = 0.05;
			for (int epoch = 0; epoch < epochs; ++epoch)
			{
				LOG(DBUG) << "---------------------Epoch: " << epoch;
				for (size_t i = 0; i < inputs.size(); ++i)
				{
					std::vector<ValuePtr> in = inputs[i];
					LOG(DBUG) << "Input:" << i << "   " << in[0]->get_val() << ", " << in[1]->get_val() << ", " << in[2]->get_val();

					std::vector<ValuePtr>& targ = targets[i];
					for (const auto& t : targ)
					{
						LOG(DBUG) << "Target " << t->get_val();
					}

					// Forward propagation
					std::vector<ValuePtr> prediction = mlp(inputs[i]);
					for (const auto& p : prediction)
					{
						LOG(DBUG) << "Prediction " << p->get_val();
					}

					// Calculate loss
					ValuePtr loss = meanSquardError(targets[i], prediction);
					LOG(DBUG) << "Loss: " << loss->get_val() << "\n";

					// Reset gradients to 0
					mlp.zero_grad();

					// Backward propagation
					loss->backward();

					// Update weights
					gradientDescent(mlp.parameters(), learningRate);

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
