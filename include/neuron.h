typedef struct neuron
{
    int bias;
    struct neuron *next;
} tNeuron;

extern tNeuron *CreateNeuron(int blias);