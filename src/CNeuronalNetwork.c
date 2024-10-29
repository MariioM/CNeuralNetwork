#include <stdio.h>
#include "../include/networkElements.h"

#define LEARNING_RATE 0.001
#define EPOCHS 10

void PrintLayerConnections(tLayer *layer, int layer_index);

int main()
{
    double celsius[] = {-40.0, -10.0, 0.0, 8.0, 15.0, 22.0, 28.0};
    double fahrenheit[] = {-40.0, 14.0, 32.0, 46.0, 59.0, 72.0, 100.0};
    int data_count = sizeof(celsius) / sizeof(celsius[0]);

    tLayer *entryLayer = CreateLayer(1);
    tLayer *hiddenLayer1 = CreateLayer(3);
    tLayer *hiddenLayer2 = CreateLayer(3);
    tLayer *outputLayer = CreateLayer(1);

    CreateConnectionsBetweenLayers(entryLayer, hiddenLayer1);
    CreateConnectionsBetweenLayers(hiddenLayer1, hiddenLayer2);
    CreateConnectionsBetweenLayers(hiddenLayer2, outputLayer);

    TrainNetwork(entryLayer, hiddenLayer1, hiddenLayer2, outputLayer, celsius, fahrenheit, data_count, EPOCHS, LEARNING_RATE);

    /*printf("Estructura de la red neuronal:\n");
    PrintLayerConnections(entryLayer, 0);
    PrintLayerConnections(hiddenLayer1, 1);
    PrintLayerConnections(hiddenLayer2, 2);
    PrintLayerConnections(outputLayer, 3);*/

    SetInput(entryLayer, -40);
    ForwardPropagation(entryLayer);
    SetLayerInputFromPreviousLayer(entryLayer, hiddenLayer1);
    ForwardPropagation(hiddenLayer1);
    SetLayerInputFromPreviousLayer(hiddenLayer1, hiddenLayer2);
    ForwardPropagation(hiddenLayer2);
    SetLayerInputFromPreviousLayer(hiddenLayer2, outputLayer);
    ForwardPropagation(outputLayer);

    printf("Salida neurona de salida: %.3f\n", outputLayer->neurons[0]->input);

    return 0;
}

void PrintLayerConnections(tLayer *layer, int layer_index)
{
    printf("Capa %d: %d neuronas\n", layer_index, layer->neuron_count);

    for (int j = 0; j < layer->neuron_count; j++)
    {
        tNeuron *neuron = layer->neurons[j];
        printf("  Neurona %d:\n", j);
        printf("    Bias: %d\n", neuron->bias);
        printf("    Conexiones salientes:\n");

        for (int k = 0; k < neuron->outgoing_count; k++)
        {
            tConnection *connection = neuron->outgoing_connections[k];
            printf("      Conexión a neurona destino en capa siguiente con peso: %.3f\n", connection->weight);
        }
    }
    printf("\n");
}
