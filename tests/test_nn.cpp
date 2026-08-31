#include <gtest/gtest.h>
#include <cstdlib>
#include "Engine.h"
#include "NN.h"

// Test de entrenamiento de la compuerta XOR usando la interfaz Network
TEST(NNTest, XORTraining) {

    // Arquitectura: 2 entradas -> 3 neuronas ocultas (tanh) -> 1 salida (sigmoid)
    Network<
            Layer<2, 3, Activation::TANH>,
            Layer<3, 1, Activation::SIGMOID>
    > net;

    float X[4][2] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };
    float Y[4][1] = {{0.0f}, {1.0f}, {1.0f}, {0.0f}};

    float lr = 0.1f;
    net.train(X, Y, 1200, lr);

    // Verificar que las predicciones aprendieron la tabla XOR
    for (int s = 0; s < 4; s++) {
        zero_grad();
        const auto& out = net.forward(X[s]);
        EXPECT_NEAR(out[0].data(), Y[s][0], 0.25f);
    }
}

