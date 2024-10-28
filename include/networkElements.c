#include "networkElements.h"
#include <stdlib.h>

tNeuron *CreateNeuron(int bias)
{
    tNeuron *new_neuron = (tNeuron *)malloc(sizeof(tNeuron));
    new_neuron->bias = bias;
    new_neuron->next = NULL;
    return new_neuron;
}