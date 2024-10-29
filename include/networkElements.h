#ifndef NETWORK_ELEMENTS_H
#define NETWORK_ELEMENTS_H

typedef struct neuron
{
    double bias;
    double output;
    double input;
    double output_error;
    struct neuron *next;
    struct connection **outgoing_connections;
    int outgoing_count;
} tNeuron;

typedef struct connection
{
    double weight;
    struct neuron *origin, *destination;
} tConnection;

typedef struct layer
{
    tNeuron **neurons;
    int neuron_count;
    struct layer *next_layer;
} tLayer;

typedef struct network
{
    tLayer *layers;
    int layer_count;
} tNetwork;

extern tNeuron *CreateNeuron();
extern tConnection *CreateConnection(tNeuron *originNeuron, tNeuron *destinationNeuron, int n);
extern tLayer *CreateLayer(int neuron_count);

extern void CreateConnectionsBetweenLayers(tLayer *layer1, tLayer *layer2);
extern void SetInput(tLayer *layer, double input);
extern void ForwardPropagation(tLayer *currentLayer);
extern void SetLayerInputFromPreviousLayer(tLayer *previousLayer, tLayer *currentLayer);

extern void CalculateOutputError(tLayer *outputLayer, double expected);
extern void Backpropagate(tLayer *currentLayer, float learning_rate);
extern void TrainNetwork(tLayer *entryLayer, tLayer *hiddenLayer1, tLayer *hiddenLayer2, tLayer *outputLayer, double *inputs, double *expected_outputs, int data_count, int epochs, float learning_rate);

#endif
