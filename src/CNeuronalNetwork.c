#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../include/networkElements.h"

#define LEARNING_RATE 0.001
#define EPOCHS 10000

int main()
{
    srand(time(NULL));

    // Training data
    double celsius[] = {-40.0, -10.0, 0.0, 8.0, 15.0, 22.0, 28.0, 35.0};
    double fahrenheit[] = {-40.0, 14.0, 32.0, 46.4, 59.0, 71.6, 82.4, 95.0};
    int data_count = sizeof(celsius) / sizeof(celsius[0]);

    // Network creation
    tLayer *entryLayer = CreateLayer(1);
    tLayer *hiddenLayer1 = CreateLayer(4);
    tLayer *outputLayer = CreateLayer(1);

    CreateConnectionsBetweenLayers(entryLayer, hiddenLayer1);
    CreateConnectionsBetweenLayers(hiddenLayer1, outputLayer);

    // Training
    printf("Training...\n");
    TrainNetwork(entryLayer, hiddenLayer1, outputLayer, celsius, fahrenheit, data_count, EPOCHS, LEARNING_RATE);

    // Tests with new values
    printf("\nPruebas con nuevos valores:\n");
    double test_temps[] = {-20.0, 5.0, 25.0, 40.0};
    for (int i = 0; i < 4; i++)
    {
        SetInput(entryLayer, test_temps[i]);
        ForwardPropagation(entryLayer, 0);
        SetLayerInputFromPreviousLayer(entryLayer, hiddenLayer1);
        ForwardPropagation(hiddenLayer1, 0);
        SetLayerInputFromPreviousLayer(hiddenLayer1, outputLayer);
        ForwardPropagation(outputLayer, 1);

        double predicted = denormalize(outputLayer->neurons[0]->output, -40, 100);
        double real = (test_temps[i] * 9.0 / 5.0) + 32;
        printf("%.1f°C -> Predicción: %.1f°F (Real: %.1f°F)\n", test_temps[i], predicted, real);
    }

    return 0;
}