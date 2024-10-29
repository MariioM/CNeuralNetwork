#include <stdio.h>
#include "../include/networkElements.h"

void PrintLayerConnections(tLayer *layer, int layer_index);

int main()
{
    tLayer *entryLayer = CreateLayer(1);
    tLayer *hiddenLayer1 = CreateLayer(3);
    tLayer *hiddenLayer2 = CreateLayer(3);
    tLayer *outputLayer = CreateLayer(1);

    CreateConnectionsBetweenLayers(entryLayer, hiddenLayer1);
    CreateConnectionsBetweenLayers(hiddenLayer1, hiddenLayer2);
    CreateConnectionsBetweenLayers(hiddenLayer2, outputLayer);

    printf("Estructura de la red neuronal:\n");
    PrintLayerConnections(entryLayer, 0);
    PrintLayerConnections(hiddenLayer1, 1);
    PrintLayerConnections(hiddenLayer2, 2);
    PrintLayerConnections(outputLayer, 3);
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
