#include "Jahley.h"

const std::string APP_NAME = "MicrogradCPP";

constexpr uint32_t INPUT_LAYER_NEURONS = 10;
constexpr uint32_t OUTPUT_LAYER_NEURONS = 1;
constexpr uint32_t NUMBER_OF_INPUTS = 20;
constexpr uint32_t HIDDEN_LAYER_NEURONS = 8;
constexpr uint32_t EPOCHS = 1000;
constexpr double LEARNING_RATE = 0.025;

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
void gradientDescent(const std::vector<ValuePtr>& params)
{
	for (auto& p : params)
	{
		double gradient = p->get_grad();

		p->set_val(p->get_val() - LEARNING_RATE * gradient);
	}
}

void fillInputs(std::vector<std::vector<ValuePtr>>& inputs)
{
	inputs.resize(NUMBER_OF_INPUTS); // Prepare the outer vector to hold NUMBER_OF_INPUTS vectors.
	for (auto& inner : inputs)
	{
		inner.resize(INPUT_LAYER_NEURONS); // Each inner vector will hold INPUT_LAYER_NEURONS ValuePtrs.

		for (auto& val : inner)
		{
			val = ExprNode::Create(generateRandomDouble(-4.0, 4.0));
		}
	}
}

void fillTargets(std::vector<std::vector<ValuePtr>>& targets) 
{
	targets.resize(NUMBER_OF_INPUTS); // Prepare the outer vector to hold NUMBER_OF_INPUTS vectors.


	for (auto& inner : targets)
	{
		inner.resize(1); // Each inner vector will hold 1 ValuePtrs.

		for (auto& val : inner)
		{
			// The tanh function outputs values in the range -1 to 1. 
			val = ExprNode::Create(generateRandomDouble(-1.0, 1.0));
		}
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
			// Create a multithreaded MLP with 3 hidden layers
			bool multiThreaded = true;
			MLP mlp(INPUT_LAYER_NEURONS, { HIDDEN_LAYER_NEURONS, HIDDEN_LAYER_NEURONS, HIDDEN_LAYER_NEURONS, OUTPUT_LAYER_NEURONS }, multiThreaded);

			std::vector<std::vector<ValuePtr>> inputs;
			fillInputs(inputs);

			std::vector<std::vector<ValuePtr>> targets;
			fillTargets(targets);

			// "Epoch" is a term used in machine learning to denote one complete pass 
			// through the entire training dataset. It's used as a measure of the number
			// of times the learning algorithm has worked through the entire training set.
			for (int epoch = 0; epoch < EPOCHS; ++epoch)
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
					gradientDescent(mlp.parameters());
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
