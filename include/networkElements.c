#include "networkElements.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>

tNeuron *CreateNeuron()
{
    tNeuron *new_neuron = (tNeuron *)malloc(sizeof(tNeuron));
    new_neuron->bias = 0;
    new_neuron->output = 0;
    new_neuron->next = NULL;
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
