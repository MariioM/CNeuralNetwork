// networkFunctions.c

#include <networkElements.h>
#include <stdlib.h>

void CreateConnectionsBetweenLayers(tLayer *layer1, tLayer *layer2)
{
    for (int i = 0; i < layer1->neuron_count; i++)
    {
        tNeuron *originNeuron = layer1->neurons[i];
        originNeuron->outgoing_connections = (tConnection **)malloc(layer2->neuron_count * sizeof(tConnection *));
        originNeuron->outgoing_count = layer2->neuron_count;

        for (int j = 0; j < layer2->neuron_count; j++)
        {
            tNeuron *destinationNeuron = layer2->neurons[j];
            originNeuron->outgoing_connections[j] = CreateConnection(originNeuron, destinationNeuron, layer1->neuron_count);
        }
    }
}