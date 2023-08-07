#include "Jahley.h"
#include <benchmark/benchmark.h>

const std::string APP_NAME = "Micrograd";

constexpr uint32_t INPUT_LAYER_NEURONS = 8;
constexpr uint32_t OUTPUT_LAYER_NEURONS = 1;
constexpr uint32_t NUMBER_OF_INPUTS = 10;
constexpr uint32_t HIDDEN_LAYER_NEURONS = 8;
constexpr uint32_t EPOCHS = 1000;
double LEARNING_RATE = 0.025;


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

static void BM_MLP_MT(benchmark::State& state) {
	for (auto _ : state) 
	{
		// Create a multitheaded MLP with 3 hidden layers
		bool multithreaded = true;
		MLP mlp(INPUT_LAYER_NEURONS, { HIDDEN_LAYER_NEURONS, HIDDEN_LAYER_NEURONS, HIDDEN_LAYER_NEURONS, OUTPUT_LAYER_NEURONS }, multithreaded);

		std::vector<std::vector<ValuePtr>> inputs;
		fillInputs(inputs);

		std::vector<std::vector<ValuePtr>> targets;
		fillTargets(targets);

		int epochs = state.range(0);
		double learningRate = 0.05;
		for (int epoch = 0; epoch < epochs; ++epoch) {
			for (size_t i = 0; i < inputs.size(); ++i) {
				// Forward propagation
				std::vector<ValuePtr> prediction = mlp(inputs[i]);

				// Calculate loss
				ValuePtr loss = meanSquardError(targets[i], prediction);

				// Reset gradients to 0
				mlp.zero_grad();

				// Backward propagation
				loss->backward();

				// Update weights
				gradientDescent(mlp.parameters());
			}
		}
	}
}

static void BM_MLP_MT_2H(benchmark::State& state) {
	for (auto _ : state)
	{

		// Create a multitheaded MLP with 2 hidden layers
		bool multithreaded = true;
		MLP mlp(INPUT_LAYER_NEURONS, { HIDDEN_LAYER_NEURONS, HIDDEN_LAYER_NEURONS,OUTPUT_LAYER_NEURONS }, multithreaded);

		std::vector<std::vector<ValuePtr>> inputs;
		fillInputs(inputs);

		std::vector<std::vector<ValuePtr>> targets;
		fillTargets(targets);

		int epochs = state.range(0);
		double learningRate = 0.05;
		for (int epoch = 0; epoch < epochs; ++epoch) {
			for (size_t i = 0; i < inputs.size(); ++i) {
				// Forward propagation
				std::vector<ValuePtr> prediction = mlp(inputs[i]);

				// Calculate loss
				ValuePtr loss = meanSquardError(targets[i], prediction);

				// Reset gradients to 0
				mlp.zero_grad();

				// Backward propagation
				loss->backward();

				// Update weights
				gradientDescent(mlp.parameters());
			}
		}
	}
}

static void BM_MLP(benchmark::State& state) {
	for (auto _ : state) 
	{
		// Create an MLP with 3 hidden layers
		MLP mlp(INPUT_LAYER_NEURONS, { HIDDEN_LAYER_NEURONS, HIDDEN_LAYER_NEURONS, HIDDEN_LAYER_NEURONS, OUTPUT_LAYER_NEURONS }, false);

		std::vector<std::vector<ValuePtr>> inputs;
		fillInputs(inputs);

		std::vector<std::vector<ValuePtr>> targets;
		fillTargets(targets);

		int epochs = state.range(0);
		
		for (int epoch = 0; epoch < epochs; ++epoch) {
			for (size_t i = 0; i < inputs.size(); ++i) {
				// Forward propagation
				std::vector<ValuePtr> prediction = mlp(inputs[i]);

				// Calculate loss
				ValuePtr loss = meanSquardError(targets[i], prediction);

				// Reset gradients to 0
				mlp.zero_grad();

				// Backward propagation
				loss->backward();

				// Update weights
				gradientDescent(mlp.parameters());
			}
		}
	}
}

static void BM_MLP_2H(benchmark::State& state) {
	for (auto _ : state)
	{
		// Create an MLP with 3 hidden layers
		MLP mlp(INPUT_LAYER_NEURONS, { HIDDEN_LAYER_NEURONS, HIDDEN_LAYER_NEURONS, OUTPUT_LAYER_NEURONS }, false);

		std::vector<std::vector<ValuePtr>> inputs;
		fillInputs(inputs);

		std::vector<std::vector<ValuePtr>> targets;
		fillTargets(targets);

		int epochs = state.range(0);

		for (int epoch = 0; epoch < epochs; ++epoch) {
			for (size_t i = 0; i < inputs.size(); ++i) {
				// Forward propagation
				std::vector<ValuePtr> prediction = mlp(inputs[i]);

				// Calculate loss
				ValuePtr loss = meanSquardError(targets[i], prediction);

				// Reset gradients to 0
				mlp.zero_grad();

				// Backward propagation
				loss->backward();

				// Update weights
				gradientDescent(mlp.parameters());
			}
		}
	}
}

// Register the function as a benchmark
BENCHMARK(BM_MLP_MT)->Arg(1000)->Arg(2000)->Arg(3000)->Arg(4000)->Arg(5000)->Unit(benchmark::kSecond);
BENCHMARK(BM_MLP)->Arg(1000)->Arg(2000)->Arg(3000)->Arg(4000)->Arg(5000)->Unit(benchmark::kSecond);
BENCHMARK(BM_MLP_MT_2H)->Arg(1000)->Arg(2000)->Arg(3000)->Arg(4000)->Arg(5000)->Unit(benchmark::kSecond);
BENCHMARK(BM_MLP_2H)->Arg(1000)->Arg(2000)->Arg(3000)->Arg(4000)->Arg(5000)->Unit(benchmark::kSecond);

class Application : public Jahley::App
{
public:
	Application() :
		Jahley::App()
	{
		int argc = 1;

		// https://stackoverflow.com/questions/2392308/c-vector-of-char-array
		// You cannot store arrays in vectors (or in any other standard library container).
		// The things that standard library containers store must be copyable and assignable,
		// and arrays are neither of these
		std::vector<char*> argv;
		char test[] = "Micrograd";
		argv.push_back(test);

		benchmark::Initialize(&argc, argv.data());
		benchmark::TimeUnit::kMillisecond; //  how to set this???
		benchmark::RunSpecifiedBenchmarks();
	}

private:
};

Jahley::App* Jahley::CreateApplication()
{
	return new Application();
}
