
inline node_id_t PoolBase::alloc(float data, node_id_t l, node_id_t r, OpType op, bool requires_grad, Role role) {
    PoolBase& pool = PoolBase::getInstance();
    ValueNode* nodes = pool.nodes;

    if (op != OpType::LEAF) {
        bool left_req = (l != ValueNode::NO_PARENT) && nodes[l].requires_grad();
        bool right_req = (r != ValueNode::NO_PARENT) && nodes[r].requires_grad();
        requires_grad = left_req || right_req;
    }
 
    nodes[pool.i].data = data;
    nodes[pool.i].grad = 0.0f;
    nodes[pool.i].prev_l = l;
    nodes[pool.i].prev_r = r;
    nodes[pool.i].init_flags(op, requires_grad, role);
    if (role == Role::PARAM) {
        pool.i_param = pool.i_param + 1;
    };
    node_id_t old_idx = pool.i;
    pool.i = pool.i + 1;
    return old_idx;
}

inline void PoolBase::reset() {
   this->i = this->i_param; 
}


inline Value::Value() {
    this->idx = 0; // constructor por defecto para los arreglos fijos
}

inline Value::Value(float data) {
    this->idx = PoolBase::alloc(data, ValueNode::NO_PARENT, ValueNode::NO_PARENT, OpType::LEAF, false, Role::ACTIVATION);
}

inline Value::Value(float data, bool requires_grad, Role role) {
    this->idx = PoolBase::alloc(data, ValueNode::NO_PARENT, ValueNode::NO_PARENT, OpType::LEAF, requires_grad, role);
}

inline Value::Value(node_id_t idx) {
    // assert( (Pool::i_act - 1) == idx);
    this->idx = idx;
}


inline float Value::data() const {
    return PoolBase::getInstance().nodes[this->idx].data;
}

inline float& Value::data() {
    return PoolBase::getInstance().nodes[this->idx].data;
}

inline float Value::grad() const {
    return PoolBase::getInstance().nodes[this->idx].grad;
}

inline float& Value::grad() {
    return PoolBase::getInstance().nodes[this->idx].grad;
}

inline void accgrad(node_id_t idx, float val) {
    if (idx != ValueNode::NO_PARENT && PoolBase::getInstance().nodes[idx].requires_grad()) {
        PoolBase::getInstance().nodes[idx].grad += val;
    }

}

inline void calculate_grad(ValueNode& node, node_id_t left_idx, node_id_t right_idx) {
    switch (node.op()) {

        case OpType::LEAF:
            // Nothing to propagate
            break;

        case OpType::TANH: {
            accgrad(left_idx, node.grad * (1.0f - pow(node.data, 2.0f)));
            break;
        }

        case OpType::SIGMOID: {
            accgrad(left_idx, node.grad * (node.data * (1.0f - node.data)) );
            break;
        }

        case OpType::RELU: {
            accgrad(left_idx, (node.data > 0.0f ? 1.0f : 0.0f) * node.grad );
            break;
        }

        case OpType::ADD: {
            // z = x + y
            // x ---
            //     |-- z -- l
            // y ---
            // dl/dx -- 
            //        |-- dl/dz
            // dl/dy --
            // chain rule: 
            //  dl/dx = dl/dz * dz/dx = dl/dz * 1
            //  dl/dy = dl/dz * dz/dy = dl/dz * 1
            // what if z = x + x?, then dz/dx = 2, loss grad is 2 * dl/dz
            // the same happens if z = y + y by simmetry
            accgrad(right_idx, node.grad);
            accgrad(left_idx, node.grad);
            break;
        }

        case OpType::SUB: {
            accgrad(right_idx, -node.grad);
            accgrad(left_idx, node.grad);
            break;
        }


        case OpType::POW: {
            // z = x^y
            // dz/dx = y * x ** (y-1)
            // dz/dy = ? y cant be a value, so its a constant then it does not change loss gradient
            float base = PoolBase::getInstance().nodes[left_idx].data;
            float exp = PoolBase::getInstance().nodes[right_idx].data;
            accgrad(left_idx, node.grad * (exp * pow(base, exp - 1) ));
            break;
        }

        case OpType::MUL: {
            // z = x * y
            // x - 
            //   |-- z -- l
            // y -
            // z(x,y) = x(u) * y(u)
            // dz/dx = dx(u)/dx * y(u) + dy(u)/dx * x = y
            // dz/dy = dx(u)/dy * y(u) + dy(u)/dy * x = x
            // what if:
            // z = x(u) = y(u) = h(u), z = h(u)^2
            // dz/dh = 2*h(u)
            float l_data = PoolBase::getInstance().nodes[left_idx].data;
            float r_data = PoolBase::getInstance().nodes[right_idx].data;
            accgrad(right_idx, node.grad * l_data);
            accgrad(left_idx, node.grad * r_data);
            break;
        }
    }
}


// inline void topo_sort(node_id_t idx, node_id_t &j, node_id_t (&result)[Pool::CAPACITY]) {
    // node_id_t stack[Pool::CAPACITY];
    // node_id_t i = 0; // next free stack position
    // node_id_t current_idx = idx;
    // stack[i] = current_idx;
    // i = i + 1;
    // while (i != 0) {
    //     current_idx = stack[i - 1];
    //     ValueNode& current = PoolBase::getInstance().nodes[current_idx]; 

    //     if (current.state == VisitState::FINISHED) {
    //         i = i - 1; // its been already included on the result array
    //         continue;
    //     }

    //     if (current.op == OpType::LEAF || current.state == VisitState::EXPANDING) {
    //         result[j] = current_idx;
    //         j = j + 1;
    //         current.state = VisitState::FINISHED;
    //         i = i - 1;

    //     } else {
    //         // at least one father
    //         current.state = VisitState::EXPANDING;

    //         if (current.prev_r != ValueNode::NO_PARENT && PoolBase::getInstance().nodes[current.prev_r].requires_grad) {
    //             stack[i] = current.prev_r;
    //             i = i + 1;
    //         }

    //         if (current.prev_l != ValueNode::NO_PARENT && PoolBase::getInstance().nodes[current.prev_l].requires_grad) {
    //             stack[i] = current.prev_l;
    //             i = i + 1;
    //         }
    //     }
    // };

    // node_id_t j = 0; // next free result position
    // node_id_t result[Pool::CAPACITY];
    // topo_sort(this->idx, j, result);

    // for (int k = j - 1; k >= 0; k--) {
    //     node_id_t inode = result[k];
    //     ValueNode& result_node = PoolBase::getInstance().nodes[inode];
    //     calculate_grad(result_node, result_node.prev_l, result_node.prev_r);
    //     result_node.state = VisitState::UNVISITED;
    // };
// }


inline void Value::backward() {
    if (!PoolBase::getInstance().nodes[this->idx].requires_grad()) {
        return;
    }
    PoolBase::getInstance().nodes[this->idx].grad = 1.0f; // dx/dx = 1

    for (int k = this->idx; k >= 0; k--) {
        ValueNode& result_node = PoolBase::getInstance().nodes[k];
        if (result_node.grad != 0.0 && result_node.op() != OpType::LEAF) {
            calculate_grad(result_node, result_node.prev_l, result_node.prev_r);
        }
    };
} 


inline Value Value::operator+(const Value& other) const {
    float result = this->data() + other.data();
    node_id_t new_idx = PoolBase::alloc(result, this->idx, other.idx, OpType::ADD, true, Role::ACTIVATION);
    return Value(new_idx);
}

inline Value Value::operator*(const Value& other) const {
    float result = this->data() * other.data();
    node_id_t new_idx = PoolBase::alloc(result, this->idx, other.idx, OpType::MUL, true, Role::ACTIVATION);
    return Value(new_idx);
}

inline Value Value::operator-(const Value& other) const {
    float result = this->data() - other.data();
    node_id_t new_idx = PoolBase::alloc(result, this->idx, other.idx, OpType::SUB, true, Role::ACTIVATION);
    return Value(new_idx);
}

inline Value Value::operator-() const {
    return *this * Value(-1.0f);
}

inline Value Value::operator/(const Value& other) const {
    return *this * (other^(-1.0f));
}

inline Value Value::operator^(float exponent) const {
    float result = pow(this->data(), exponent);
    node_id_t exp_idx = PoolBase::alloc(exponent, ValueNode::NO_PARENT, ValueNode::NO_PARENT, OpType::LEAF, false, Role::ACTIVATION);
    node_id_t new_idx = PoolBase::alloc(result, this->idx, exp_idx, OpType::POW, true, Role::ACTIVATION);
    return Value(new_idx);
}


inline Value tanh(const Value& x) {
    node_id_t new_idx = PoolBase::alloc(::tanh(x.data()), x.idx, ValueNode::NO_PARENT, OpType::TANH, true, Role::ACTIVATION);
    return Value(new_idx);
}

inline Value relu(const Value& x) {
    float val = (0.0f > x.data()) ? 0.0f : x.data();
    node_id_t new_idx = PoolBase::alloc(val, x.idx, ValueNode::NO_PARENT, OpType::RELU, true, Role::ACTIVATION);
    return Value(new_idx);
}

inline Value sigmoid(const Value& x) {
    float val = 1.0f / (1.0f + exp(-x.data()));
    node_id_t new_idx = PoolBase::alloc(val, x.idx, ValueNode::NO_PARENT, OpType::SIGMOID, true, Role::ACTIVATION);
    return Value(new_idx);
}

inline Value mse(const float& exp, const Value& pred) {
    // 3
    Value val = (Value(exp) - pred);
    return val * val;
}

inline void zero_grad() {
    PoolBase& pool = PoolBase::getInstance();
    pool.reset();
    for (int i = 0; i < pool.i_param; i++) {
        pool.nodes[i].grad = 0.0f;        
    }
}


inline void sgd(float step) {
    PoolBase& pool = PoolBase::getInstance();
    for (int i = 0; i < pool.i_param; i++) {
       pool.nodes[i].data -=  step * pool.nodes[i].grad; 
    }
}
