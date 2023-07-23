#pragma once

// attempt at c++ version of micrograd
// https://github.com/karpathy/micrograd

// This class was created with some help from ChatGPT4
using ValuePtr = std::shared_ptr<class ExprNode>;

// The ExprNode(Expression Node) class is enabled to manage shared_ptr instances of itself
class ExprNode : public std::enable_shared_from_this<class ExprNode>
{
public:
	// Factory method for creating instances of ExprNode
	static ValuePtr Create(double data, std::initializer_list<ValuePtr> children = {}, std::string op = "")
	{
		// Create a new instance of ExprNode
		ValuePtr instance = std::make_shared<ExprNode>(data);

		// Assign provided children and operation
		instance->_prev = children;
		instance->_op = op;

		// Initialize _backward with an empty function
		instance->_backward = [] {};

		return instance;
	}

	// Constructor that takes initial data and initializes grad to 0
	ExprNode(double data) :
		data(data), grad(0.0) {}

	~ExprNode()
	{
		LOG(DBUG) << "NODE is destroyed ";
	}

	// Operator overload for addition with another ValuePtr
	ValuePtr operator+ (ValuePtr other)
	{
		// If other is null, create a new ExprNode with data 0
		if (!other) other = Create(0);

		// Create a new ExprNode which is the sum of the current and other
		auto out = Create(this->data + other->data, { shared_from_this(), other }, "+");

		// Set up _backward function to compute and store gradients
		out->_backward = [this, other, out]()
		{
			this->grad += out->grad;
			other->grad += out->grad;
		};
		return out;
	}

	// Operator overload for subtraction with another ValuePtr
	ValuePtr operator- (ValuePtr other)
	{
		return *this + -(*other);
	}

	// Operator overload for addition with a double
	ValuePtr operator+ (double val)
	{
		// Create a new ExprNode with data val
		ValuePtr other = Create(val);

		// Process as in the previous method
		auto out = Create(this->data + other->data, { shared_from_this(), other }, "+");
		out->_backward = [this, other, out]()
		{
			this->grad += out->grad;
			other->grad += out->grad;
		};
		return out;
	}

	// Operator overload for multiplication with another ValuePtr
	ValuePtr operator* (ValuePtr other)
	{
		if (!other) other = std::make_shared<ExprNode>(1);
		auto out = Create(this->data * other->data, { shared_from_this(), other }, "*");
		out->_backward = [this, other, out]()
		{
			this->grad += other->data * out->grad;
			other->grad += this->data * out->grad;
		};
		return out;
	}

	// Operator overload for division with a double
	ValuePtr operator/ (double val)
	{
		if (val == 0.0)
		{
			throw std::invalid_argument("Division by zero is not allowed");
		}
		return *this * (1.0 / val);
	}

	// Operator overload for division with another ValuePtr
	ValuePtr operator/ (ValuePtr other)
	{
		if (!other || other->get_val() == 0.0)
		{
			throw std::invalid_argument("Division by zero is not allowed");
		}
		return *this * other->pow(-1);
	}

	// Operator overload for multiplication with a double
	ValuePtr operator* (double val)
	{
		ValuePtr other = Create(val);
		auto out = Create(this->data * other->data, { shared_from_this(), other }, "*");
		out->_backward = [this, other, out]()
		{
			this->grad += other->data * out->grad;
			other->grad += this->data * out->grad;
		};
		return out;
	}

	// Power operation
	ValuePtr pow(double other)
	{
		auto out = Create(std::pow(this->data, other), { shared_from_this() }, "^");
		out->_backward = [this, other, out]()
		{
			this->grad += (other * std::pow(this->data, other - 1)) * out->grad;
		};
		return out;
	}

	// Negation operator
	ValuePtr operator-()
	{
		ValuePtr negativeOne = Create(-1.0);
		return (*this) * negativeOne;
	}

	// Backward propagation
	void backward()
	{
		// Topological sort to find execution order
		std::vector<ValuePtr> topo;
		std::set<ValuePtr> visited;
		std::function<void(ValuePtr)> build_topo = [&](ValuePtr v)
		{
			if (visited.find(v) == visited.end())
			{
				visited.insert(v);
				for (auto& child : v->_prev)
				{
					build_topo(child);
				}
				topo.push_back(v);
			}
		};
		build_topo(shared_from_this());

		// Execute _backward in topological order
		this->grad = 1.0;
		for (auto it = topo.rbegin(); it != topo.rend(); ++it)
		{
			(*it)->_backward();
		}
	}

	// ReLU activation function
	ValuePtr relu()
	{
		auto out = Create(this->data < 0.0 ? 0.0 : this->data, { shared_from_this() }, "ReLU");

		out->_backward = [this, out]()
		{
			double t = out->data > 0.0 ? 1.0 : 0.0;
			this->grad = this->grad + out->get_grad() * t;
		};

		return out;
	}

	// Getters for data and grad
	double get_val()
	{
		return data;
	}
	double get_grad()
	{
		return grad;
	}
	void set_grad(double val)
	{
		grad = val;
	}
	void set_val(double val)
	{
		data = val;
	}

private:
	double data;                     // The data held by the ExprNode
	double grad;                     // The gradient of the ExprNode
	std::string _op;                 // The operation that produced this ExprNode
	std::vector<ValuePtr> _prev;     // The previous Values that this ExprNode depends on
	std::function<void()> _backward; // The function to propagate gradients back through this ExprNode
};

// The Module class is an abstract base class that represents a component of a neural network.
// It includes methods for handling the parameters of the component (such as the weights and biases of neurons),
// and for performing backpropagation.
class Module
{
public:
	// The 'zero_grad' member function sets the gradients of all parameters in the module to zero.
	// This is typically used at the start of a new round of backpropagation.
	virtual void zero_grad()
	{
		for (auto& p : this->parameters())
		{
			p->set_grad(0);
		}
	}

	// The 'parameters' member function returns a vector containing all the parameters of the module.
	// In the base class, this function just returns an empty vector. Subclasses (like Neuron, Layer, and MLP)
	// will override this method to return the actual parameters of the module.
	virtual std::vector<ValuePtr> parameters()
	{
		return {};
	}
};

// The Neuron class represents a single neuron in a neural network.
// It is a subclass of the Module class.
class Neuron : public Module, public HasId
{
private:
	std::vector<ValuePtr> weights; // The 'weights' member holds the weights of the neuron's inputs.
	// Each element in this vector corresponds to the weight of a particular input.

	ValuePtr bias; // The 'bias' member represents the bias of the neuron.
	// This is an additional parameter added to the weighted sum of the neuron's inputs,
	// which shifts the output of the neuron's activation function.

	bool nonlin; // The 'nonlin' member is a flag that determines whether a non-linear activation function
	// (in this case, ReLU) should be applied to the output of the neuron.

public:
	// The constructor takes the number of inputs ('inputCount') and a boolean indicating whether non-linearity should be applied.
	// The weights of the neuron are initialized with random values in the range [-1.0, 1.0],
	// and the bias is initialized to 0.
	Neuron(int inputCount, bool nonlin = true) :
		nonlin(nonlin)
	{
		weights.reserve(inputCount);

		for (int i = 0; i < inputCount; ++i)
		{
			// using generateRandomDouble fixes the intermittent failure problems
			// because it produces the same random number set on every run
			double weight = generateRandomDouble();
			weights.push_back(ExprNode::Create(weight));
		}

		bias = ExprNode::Create(0.001);
	}

	// The function call operator is overloaded to compute the output of the neuron given its inputs.
	// It computes the weighted sum of the inputs and bias,
	// and then applies the ReLU activation function if 'nonlin' is true.
	ValuePtr operator() (const std::vector<ValuePtr>& inputs)
	{
		assert(inputs.size() == weights.size());

		ValuePtr activation = bias;
		for (int i = 0; i < weights.size(); ++i)
		{
			ValuePtr t = *weights[i] * inputs[i];

			activation = *activation + t;

		}
		return nonlin ? activation->relu() : activation;
	}

	// The 'parameters' member function returns a vector containing all the parameters (weights and bias) of the neuron.
	std::vector<ValuePtr> parameters() override
	{
		std::vector<ValuePtr> params = weights;
		params.push_back(bias);
		return params;
	}
};

// The Layer class represents a layer in a neural network. It is a subclass of the Module class.
class Layer : public Module
{
private:
	uint32_t id = 0;
	std::vector<Neuron> neurons; // The 'neurons' member holds the set of neurons that make up the layer.
	// Each neuron is an instance of the Neuron class.

public:
	// The constructor takes the number of input and output neurons ('neuronsIn' and 'neuronsOut', respectively).
	// It initializes the layer by creating 'neuronsOut' neurons, each with 'neuronsIn' inputs.
	Layer(int neuronsIn, int neuronsOut, uint32_t id)
	{
		this->id = id;
		for (int i = 0; i < neuronsOut; ++i)
		{
			neurons.push_back(Neuron(neuronsIn));
		}
	}

	// The function call operator is overloaded to compute the output of the layer given its inputs.
	// It applies each neuron in the layer to the input, and collects the results into a vector.
	std::vector<ValuePtr> operator() (const std::vector<ValuePtr>& inputs)
	{
		std::vector<ValuePtr> out;
		out.reserve(neurons.size());

		for (auto& n : neurons)
		{
			out.push_back(n(inputs));
		}
		return out;
	}

	// The 'parameters' member function returns a vector containing all the parameters (weights and biases)
	// of the neurons in the layer.
	std::vector<ValuePtr> parameters() override
	{
		std::vector<ValuePtr> params;
		for (auto& n : neurons)
		{
			auto n_params = n.parameters();
			params.insert(params.end(), n_params.begin(), n_params.end());
		}
		return params;
	}
};

// The MLP (Multilayer Perceptron) class represents a fully connected neural network,
// composed of multiple layers. It is a subclass of the Module class.
class MLP : public Module
{
private:
	std::vector<Layer> layers; // The 'layers' member holds the sequence of layers that make up the network.

public:
	// The constructor takes the number of input neurons ('inputNeuronCount') and a vector that specifies the number of neurons
	// in each layer ('neuronsPerLayer'). It initializes the network by creating a sequence of layers,
	// each with the appropriate number of input and output neurons.
	MLP(int inputNeuronCount, std::vector<int> neuronsPerLayer)
	{
		int sz_in = inputNeuronCount;
		for (int i = 0; i < neuronsPerLayer.size(); ++i)
		{
			layers.push_back(Layer(sz_in, neuronsPerLayer[i], layers.size()));
			sz_in = neuronsPerLayer[i];
		}
	}

	// The function call operator is overloaded to compute the output of the network given its inputs.
	// It applies each layer in the network to the output of the previous layer, and returns the final output.
	std::vector<ValuePtr> operator() (const std::vector<ValuePtr>& inputs)
	{
		std::vector<ValuePtr> out = inputs;
		for (auto& layer : layers)
		{
			out = layer(out);
		}
		return out;
	}

	// The 'parameters' member function returns a vector containing all the parameters (weights and biases)
	// of the neurons in the network. It does this by concatenating the parameters from each layer.
	std::vector<ValuePtr> parameters() override
	{
		std::vector<ValuePtr> params;
		for (auto& layer : layers)
		{
			const auto& layer_params = layer.parameters();
			params.insert(params.end(), layer_params.begin(), layer_params.end());
		}
		return params;
	}
};
