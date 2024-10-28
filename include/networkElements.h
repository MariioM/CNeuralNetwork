
typedef struct neuron
{
    int bias;
    double output;
    double input;
    struct neuron *next;
} tNeuron;

typedef struct connection
{
    double weight;
    struct neuron *origin, *destination;
} tConnection;

typedef struct layer
{
    tNeuron *neurons; // Array o lista de neuronas en la capa
    int neuron_count; // Número de neuronas en la capa
} tLayer;

typedef struct network
{
    tLayer *layers;  // Array o lista de capas en la red
    int layer_count; // Número de capas
} tNetwork;

extern tNeuron *CreateNeuron();
extern tConnection *CreateConnection();