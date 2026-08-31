#include <gtest/gtest.h>
#include "Engine.h"

// test a implementar:
// grafo de operaciones con un nodo que es padre de dos ramas a la vez


// Neurona
TEST(EngineTest, Neurona) {
    Pool<128> pool;
    Value x1(1.0f, true);
    Value x2(1.0f, true);

    Value w1(0.0f, true);
    Value w2(1.0f, true);

    Value expected(0.0f);

    Value pred = tanh(x1*w1 + x2*w2);
    Value loss = (expected - pred)^2.0f;

    loss.backward();

    EXPECT_NEAR(pred.data(), 0.761594f, 1e-6f);
    EXPECT_NEAR(loss.data(), 0.580026f, 1e-6f);
    EXPECT_NEAR(x2.grad(), 0.639701f, 1e-5f);
}


// Test de creacion basica
TEST(EngineTest, CreacionValue) {
    Pool<128> pool;
    Value a(5.0f);
    EXPECT_FLOAT_EQ(a.data(), 5.0f);
    EXPECT_FLOAT_EQ(a.grad(), 0.0f);
}

// Tests del Forward pass
TEST(EngineTest, ForwardSuma) {
    Pool<128> pool;
    Value a(2.0f);
    Value b(3.0f);
    Value c = a + b;
    EXPECT_FLOAT_EQ(c.data(), 5.0f);
}

TEST(EngineTest, ForwardMultiplicacion) {
    Pool<128> pool;
    Value a(3.0f);
    Value b(4.0f);
    Value c = a * b;
    EXPECT_FLOAT_EQ(c.data(), 12.0f);
}

// Tests del Backward pass
TEST(EngineTest, BackwardSuma) {
    Pool<128> pool;
    Value a(2.0f, true);
    Value b(3.0f, true);
    Value c = a + b;
    c.grad() = 1.0f;
    c.backward();

    EXPECT_FLOAT_EQ(a.grad(), 1.0f);
    EXPECT_FLOAT_EQ(b.grad(), 1.0f);
}

TEST(EngineTest, BackwardMultiplicacion) {
    Pool<128> pool;
    Value x(2.0f, true);
    Value y(3.0f, true);
    Value z = x * y;
    z.grad() = 1.0f;
    z.backward();

    EXPECT_FLOAT_EQ(x.grad(), 3.0f);
    EXPECT_FLOAT_EQ(y.grad(), 2.0f);
}

TEST(EngineTest, BackwardMismoNodo) {
    Pool<128> pool;
    Value x(4.0f, true);
    Value z = x * x;
    z.grad() = 1.0f;
    z.backward();

    // d/dx (x^2) = 2 * x = 8.0f
    EXPECT_FLOAT_EQ(x.grad(), 8.0f);
}


TEST(EngineTest, FallaSobrescrituraGradiente) {
    Pool<128> pool;
    Value a(2.0f, true);
    Value cte1(3.0f);
    Value cte2(4.0f);
    /*
        cte1 ---      
               |-- b --
           a ---       |-- f
               |-- c --
        cte2 ---
    */
    Value b = a + cte1; // b = a + 3 -> derivada db/da = 1.0
    Value c = a * cte2; // c = a * 4 -> derivada dc/da = 4.0
    Value f = c + b;
    f.grad() = 1.0f;
    f.backward();
    EXPECT_FLOAT_EQ(a.grad(), 5.0f);
}

TEST(EngineTest, FallaGrafoProfundoSinTopologicalSort) {
    Pool<128> pool;
    // Expresion compuesta: d = (a + b) * c
    Value a(2.0f, true);
    Value b(3.0f, true);
    Value c(4.0f);

    Value suma = a + b;       // suma = 2 + 3 = 5
    Value d = suma * c;       // d = 5 * 4 = 20

    d.grad() = 1.0f;
    d.backward();

    // d = (a + b) * c  =>  dd/da = c = 4.0f,  dd/db = c = 4.0f
    EXPECT_FLOAT_EQ(suma.grad(), 4.0f);
    EXPECT_FLOAT_EQ(a.grad(), 4.0f);
    EXPECT_FLOAT_EQ(b.grad(), 4.0f);
}

