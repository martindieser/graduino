template <size_t N>
Neuron<N>::Neuron() {
	for (int i = 0; i <= N; i++) {
		float r = (((float) rand()/(float)RAND_MAX) * 2.0f) - 1.0f;
		w[i] = Value(r, true, Role::PARAM);
	}
	// w[N] = Value(0.0f, true, Role::PARAM);
}


template <size_t N>
const Value (&Neuron<N>::parameters() const)[N + 1] {
    return w;
}

template <size_t N>
Value Neuron<N>::operator()(const float (&input)[N]) const {
	Value result(this->w[0] * input[0]);
    for (int i = 1; i < N; ++i) {
        result = result + this->w[i] * Value(input[i]);
    }
    result = result + this->w[N]; // se reserva N para la ordenada

    return result;
}

template <size_t N>
Value Neuron<N>::operator()(const Value (&input)[N]) const {
	Value result = input[0] * this->w[0];
    for (int i = 1; i < N; i++) {
        result = result + this->w[i] * input[i];
    }
    result = result + this->w[N]; 
    return result;
}


template <size_t N, size_t T, Activation act>
Layer<N, T, act>::Layer() {

}

template <size_t N, size_t T, Activation act>
const Value (&Layer<N, T, act>::parameters() const)[T * (N + 1)] {
	for (int i = 0; i < T; i++) {
		const Value (&n)[N+1] = neurons[i].parameters();
		for (int j = 0; j < (N+1); j++) {
			this->params[i*(N+1)+j] = n[j];
		}
	}
	return this->params;
}


Value apply_activation(Activation act, Value val) {
    switch(act) {
		case Activation::TANH:
	    	return tanh(val);
		case Activation::SIGMOID:
	    	return sigmoid(val);    	
	    case Activation::RELU:
	    	return relu(val);
	    default:
	    	return val;
    }
}

template <size_t N, size_t T, Activation act>
const Value (&Layer<N, T, act>::operator()(const float (&input)[N]) const)[T] {
	Value v_input[N];
	for (size_t i = 0; i < N; i++) {
		v_input[i] = Value(input[i]);
	}
	return (*this)(v_input);
}

template <size_t N, size_t T, Activation act>
const Value (&Layer<N, T, act>::operator()(const Value (&input)[N]) const)[T] {
    for (int i = 0; i < T; i++) {
    	const Neuron<N>& n = neurons[i];
        output[i] = apply_activation(act, n(input));
    }
    return output;
}


template <typename... Layers>
template <size_t SAMPLES>
void Network<Layers...>::train(
		   const float (&X)[SAMPLES][IN_SIZE],
		   const float (&Y)[SAMPLES][OUT_SIZE],
		   int epochs,
		   float step) {

	PoolBase::bind(&pool);

	for (int epoch = 1; epoch <= epochs; epoch++) {
		for (size_t s = 0; s < SAMPLES; s++) {
			zero_grad();
			const auto& out = forward(X[s]);
			Value loss = mse(Y[s], out);
			loss.backward();
			sgd(step);
		}
	}
}

