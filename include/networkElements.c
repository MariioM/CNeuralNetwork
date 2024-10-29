// networkElements.c

#include "networkElements.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>

tNeuron *CreateNeuron()
{
    tNeuron *new_neuron = (tNeuron *)malloc(sizeof(tNeuron));
    new_neuron->bias = 0;
    new_neuron->output = 0;
    new_neuron->input = 0;
    new_neuron->next = NULL;
    new_neuron->outgoing_connections = NULL;
    return new_neuron;
}

tConnection *CreateConnection(tNeuron *originNeuron, tNeuron *destinationNeuron, int n)
{
    tConnection *new_connection = (tConnection *)malloc(sizeof(tConnection));
    new_connection->origin = originNeuron;
    new_connection->destination = destinationNeuron;

    // Calculation of the initial weight
    float random = (float)rand() / RAND_MAX;
    float standardDeviation = sqrt(2.0 / n);
    new_connection->weight = -standardDeviation + random * (standardDeviation - (-standardDeviation));

    return new_connection;
}

tLayer *CreateLayer(int neuron_count)
{
    tLayer *new_layer = (tLayer *)malloc(sizeof(tLayer));
    new_layer->neuron_count = neuron_count;
    new_layer->neurons = (tNeuron **)malloc(neuron_count * sizeof(tNeuron *));
    new_layer->next_layer = NULL;

    for (int i = 0; i < neuron_count; i++)
    {
        new_layer->neurons[i] = CreateNeuron();
    }
    return new_layer;
}
