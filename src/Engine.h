#ifndef GRADUINO_ENGINE
#define GRADUINO_ENGINE

#include<GraduinoConfig.h>

#include<stddef.h>
#include<assert.h>
#include<math.h>

enum OpType : uint8_t { LEAF = 0, ADD, POW, MUL, TANH, RELU, SIGMOID, SUB };
enum Role { PARAM = 0, ACTIVATION = 1 };

struct ValueNode {
    float data; // 4 bytes
    float grad; // 4 bytes
    node_id_t prev_l; // 1 byte?
    node_id_t prev_r; // 1 byte?
    uint8_t flags; // 1 byte

    // VisitState state;
    static constexpr node_id_t NO_PARENT = static_cast<node_id_t>(-1);

    static constexpr uint8_t MASK_REQ_GRAD = 0x01; // 0000 0001
    static constexpr uint8_t MASK_ROLE = 0x02; // 0000 0010
    static constexpr uint8_t SHIFT_OP = 2; // bits 2..7 


    inline bool requires_grad() const {
        return flags & MASK_REQ_GRAD;
    }
    inline void set_requires_grad(bool val) {
        if (val) {
            flags = flags | MASK_REQ_GRAD;
        } else {
            flags = flags & ~MASK_REQ_GRAD;
        };
    }

    inline Role role() const {
        return (flags & MASK_ROLE) ? Role::ACTIVATION : Role::PARAM;
    }

    inline OpType op() const {
        return static_cast<OpType>(flags >> SHIFT_OP);
    }

    inline void init_flags(OpType o, bool req_grad, Role r) {
        flags = (static_cast<uint8_t>(o) << SHIFT_OP) |
                (req_grad ? MASK_REQ_GRAD : 0) |
                (r == Role::ACTIVATION ? MASK_ROLE : 0);
    }
};



class PoolBase {
    public:
        ValueNode* nodes;
        size_t capacity;
        node_id_t i_param;
        node_id_t i;

        // static inline PoolBase* active_pool = nullptr;
        static PoolBase*& getActivePoolPtr() {
            static PoolBase* active = nullptr;
            return active;
        }

        static void bind(PoolBase* p) { getActivePoolPtr() = p;}
        static PoolBase& getInstance() { return *getActivePoolPtr();}
        static node_id_t alloc(float data, node_id_t l, node_id_t r, OpType op,
                             bool requires_grad = false, Role role = Role::ACTIVATION);

        void reset();
};

template <size_t CAPACITY>
class Pool : public PoolBase {
    private:
        ValueNode storage[CAPACITY];
    public:
        Pool() {
            this->nodes = storage;
            this->capacity = CAPACITY;
            this->i = 0; 
            this->i_param = 0;
            PoolBase::bind(this);
        }
};



class Value {
    /* stores a single scalar value and its gradient */
    public:
        node_id_t idx;
        Value();
        Value(float data);
        Value(float data, bool requires_grad, Role role = Role::ACTIVATION);
        Value(node_id_t idx);

        Value operator+(const Value& other) const;
        Value operator-(const Value& other) const;
        Value operator-() const;
        Value operator^(float exponent) const; // potencia
        Value operator*(const Value& other) const;
        Value operator/(const Value& other) const;

        void backward();

        float data() const;
        float& data();
        float grad() const; 
        float& grad();
    private:
};

// funciones de activación
Value tanh(const Value& x);
Value relu(const Value& x);
Value sigmoid(const Value& x);


// medidas de loss
Value mse(const float& exp, const Value& pred);

template <size_t T>
Value mse(const float (&exp)[T], const Value (&pred)[T] ) {
    Value total(mse(exp[0], pred[0]));
    for (int i = 1; i < T; i++) {
        total = total + mse(exp[i], pred[i]);
    };
    // 3 * T  + 4
    return total / Value((float) T);
};

void zero_grad();
void sgd(float step);


#include<Engine.tpp>

#endif