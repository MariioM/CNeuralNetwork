#include <networkElements.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define MAX_WEIGHT 1.0

double relu(double x)
{
    return x > 0 ? x : 0;
}

double relu_derivative(double x)
{
    return x > 0 ? 1 : 0;
}

double normalize(double value, double min, double max)
{
    return (value - min) / (max - min);
}

double denormalize(double normalized_value, double min, double max)
{
    return normalized_value * (max - min) + min;
}

void CreateConnectionsBetweenLayers(tLayer *layer1, tLayer *layer2)
{
    for (int i = 0; i < layer1->neuron_count; i++)
    {
        tNeuron *originNeuron = layer1->neurons[i];
        originNeuron->outgoing_connections = (tConnection **)malloc(layer2->neuron_count * sizeof(tConnection *));
        originNeuron->outgoing_count = layer2->neuron_count;

        for (int j = 0; j < layer2->neuron_count; j++)
        {
            originNeuron->outgoing_connections[j] = CreateConnection(originNeuron, layer2->neurons[j], layer1->neuron_count);
        }
    }
}

void SetInput(tLayer *layer, double input)
{
    // Normalice (aprox range)
    layer->neurons[0]->input = normalize(input, -40, 100);
    layer->neurons[0]->output = layer->neurons[0]->input;
}

void ForwardPropagation(tLayer *currentLayer, int is_output_layer)
{
    for (int i = 0; i < currentLayer->neuron_count; i++)
    {
        tNeuron *neuron = currentLayer->neurons[i];
        if (!is_output_layer)
        {
            neuron->output = relu(neuron->input + neuron->bias);
        }
        else
        {
            neuron->output = neuron->input + neuron->bias;
        }
    }
}

void SetLayerInputFromPreviousLayer(tLayer *previousLayer, tLayer *currentLayer)
{
    for (int i = 0; i < currentLayer->neuron_count; i++)
    {
        tNeuron *currentNeuron = currentLayer->neurons[i];
        currentNeuron->input = 0.0;

        for (int j = 0; j < previousLayer->neuron_count; j++)
        {
            tNeuron *prevNeuron = previousLayer->neurons[j];
            for (int k = 0; k < prevNeuron->outgoing_count; k++)
            {
                tConnection *connection = prevNeuron->outgoing_connections[k];
                if (connection->destination == currentNeuron)
                {
                    currentNeuron->input += prevNeuron->output * connection->weight;
                }
            }
        }
    }
}

void CalculateOutputError(tLayer *outputLayer, double expected)
{
    tNeuron *neuron = outputLayer->neurons[0];
    double normalized_expected = normalize(expected, -40, 100);
    neuron->output_error = normalized_expected - neuron->output;
}

void Backpropagate(tLayer *currentLayer, float learning_rate)
{
    for (int i = 0; i < currentLayer->neuron_count; i++)
    {
        tNeuron *neuron = currentLayer->neurons[i];

        if (currentLayer->next_layer != NULL)
        {
            double error = 0.0;
            for (int j = 0; j < neuron->outgoing_count; j++)
            {
                tConnection *connection = neuron->outgoing_connections[j];
                error += connection->weight * connection->destination->output_error;
            }
            neuron->output_error = error * relu_derivative(neuron->input + neuron->bias);
        }

        // Update weigths
        for (int j = 0; j < neuron->outgoing_count; j++)
        {
            tConnection *connection = neuron->outgoing_connections[j];
            double delta = learning_rate * connection->destination->output_error * neuron->output;
            connection->weight += delta;

            if (connection->weight > MAX_WEIGHT)
            {
                connection->weight = MAX_WEIGHT;
            }
            else if (connection->weight < -MAX_WEIGHT)
            {
                connection->weight = -MAX_WEIGHT;
            }
        }

        neuron->bias += learning_rate * neuron->output_error;
    }
}

void TrainNetwork(tLayer *entryLayer, tLayer *hiddenLayer1, tLayer *outputLayer, double *inputs, double *expected_outputs, int data_count, int epochs, float learning_rate)
{

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        double total_error = 0.0;

        for (int i = 0; i < data_count; i++)
        {
            // Forward pass
            SetInput(entryLayer, inputs[i]);
            ForwardPropagation(entryLayer, 0);
            SetLayerInputFromPreviousLayer(entryLayer, hiddenLayer1);
            ForwardPropagation(hiddenLayer1, 0);
            SetLayerInputFromPreviousLayer(hiddenLayer1, outputLayer);
            ForwardPropagation(outputLayer, 1);

            // Calcular y acumular error
            double predicted = denormalize(outputLayer->neurons[0]->output, -40, 100);
            double error = fabs(expected_outputs[i] - predicted);
            total_error += error;

            // Backpropagation
            CalculateOutputError(outputLayer, expected_outputs[i]);
            Backpropagate(outputLayer, learning_rate);
            Backpropagate(hiddenLayer1, learning_rate);
        }
        double avg_error = total_error / data_count;
        printf("Epoch %d: Error = %.2f\n", epoch + 1, avg_error);
    }
}