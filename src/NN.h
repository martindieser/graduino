#ifndef GRADUINO_NN
#define GRADUINO_NN

#include <Engine.h>
#include <stdlib.h>


enum class Activation {NONE, TANH, SIGMOID, RELU};

template <size_t N>
class Neuron {
	public:
	    Neuron();
	    const Value (&parameters() const)[N+1];
	    Value operator()(const float (&input)[N]) const;
	    Value operator()(const Value (&input)[N]) const;

	private:
	    Value w[N+1];
};



template <size_t N, size_t T, Activation act = Activation::NONE>
class Layer {
	public:
		static constexpr size_t INPUTS = N;
		static constexpr size_t OUTPUTS = T;

		static constexpr size_t PARAMS = T * (N+1);
		static constexpr size_t F_ACTIVATION_NODE = (act == Activation::NONE) ? 0 : 1;
		static constexpr size_t FORWARD_NODES = T * (2*N+F_ACTIVATION_NODE);

	    Layer();
	    const Value (&parameters() const)[T * (N + 1)];
	    const Value (&operator()(const float (&input)[N]) const) [T];
	    const Value (&operator()(const Value (&input)[N]) const) [T];

	private:
		mutable Value params[T * (N + 1)];
	    Neuron<N> neurons[T];
	    mutable Value output[T];
};

// Metaprogramación
// Usamos templates para calcular la suma de nodos totales en tiempo de compilación con (constexpr)


// Definición de plantilla con cantidad variable de tipos
template <size_t... Values> struct Sum;

// Base Case
template <> struct Sum<> {static constexpr size_t value = 0;};

// Recursive Case
template <size_t First, size_t... Rest>
struct Sum<First, Rest...> {
	static constexpr size_t value = First + Sum<Rest...>::value;
};


// Obtener primera capa, la Tail seria la segunda capa aca, el compilador descarta las otras
template <typename Head, typename... Tail>
struct FirstLayer { using type = Head; };

// Definición de plantilla con cantidad variable de tipos
template <typename... Layers> struct LastLayer;

// Mismo truco que con la suma...
// Caso base
template <typename Single> struct LastLayer<Single> { using type = Single; };

// Caso recursivo
template <typename Head, typename Second, typename... Rest>
struct LastLayer<Head, Second, Rest...> {
	using type = typename LastLayer<Second, Rest...>::type;
};

// Cadena de Capas
template <typename... Layers>
struct LayerChain;


// Caso base
template <typename Last>
struct LayerChain<Last> {
	Last current;

	template <typename InType>
	auto forward(const InType& input) -> decltype(current(input)) {
		return current(input);
	}
};

// Caso recursivo
template <typename Current, typename... Rest>
struct LayerChain<Current, Rest...> {
	Current current;
	LayerChain<Rest...> rest;

	template <typename InType>
	auto forward (const InType& input) -> decltype(rest.forward(current(input))) {
		return rest.forward(current(input));
	}
};



template <typename... Layers>
class Network {
public:
	using input_layer = typename FirstLayer<Layers...>::type;
	using output_layer = typename LastLayer<Layers...>::type;

	static constexpr size_t IN_SIZE = input_layer::INPUTS;
	static constexpr size_t OUT_SIZE = output_layer::OUTPUTS;

	static constexpr size_t TOTAL_PARAMS = Sum<Layers::PARAMS...>::value;
	static constexpr size_t TOTAL_FORWARD = Sum<Layers::FORWARD_NODES...>::value;
	static constexpr size_t LOSS_NODES = OUT_SIZE * 3 + (OUT_SIZE - 1) + 4;

	static constexpr size_t TOTAL_CAPACITY = TOTAL_PARAMS + IN_SIZE + TOTAL_FORWARD + LOSS_NODES  ;

	Pool<TOTAL_CAPACITY> pool;
	LayerChain<Layers...> layers;


	template <typename InType>
	auto forward(const InType& input) -> decltype(layers.forward(input)) {
		PoolBase::bind(&pool);
		return layers.forward(input);
	}

	template <size_t SAMPLES>
	void train(const float (&X)[SAMPLES][IN_SIZE],
			   const float (&Y)[SAMPLES][OUT_SIZE],
			   int epochs,
			   float step);

	private:
		mutable float out_buffer[OUT_SIZE];
};

#include<NN.tpp>
#endif