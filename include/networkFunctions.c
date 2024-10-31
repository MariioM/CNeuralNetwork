// networkFunctions.c
#include <networkElements.h>
#include <stdlib.h>
#include <stdio.h>
void CreateConnectionsBetweenLayers(tLayer *layer1, tLayer *layer2)
{
    for (int i = 0; i < layer1->neuron_count; i++)
    {
        tNeuron *originNeuron = layer1->neurons[i];

        originNeuron->outgoing_connections = (tConnection **)malloc(layer2->neuron_count * sizeof(tConnection *));
        if (originNeuron->outgoing_connections == NULL)
        {
            perror("Error al asignar memoria para conexiones salientes");
            exit(EXIT_FAILURE);
        }

        originNeuron->outgoing_count = layer2->neuron_count;

        for (int j = 0; j < layer2->neuron_count; j++)
        {
            tNeuron *destinationNeuron = layer2->neurons[j];
            originNeuron->outgoing_connections[j] = CreateConnection(originNeuron, destinationNeuron, layer1->neuron_count);
        }
    }
}

void SetInput(tLayer *layer, double input)
{
    for (int i = 0; i < layer->neuron_count; i++)
    {
        layer->neurons[i]->input = input;
        layer->neurons[i]->output = input;
    }
}

void ForwardPropagation(tLayer *currentLayer)
{
    for (int i = 0; i < currentLayer->neuron_count; i++)
    {
        tNeuron *neuron = currentLayer->neurons[i];
        neuron->output = 0.0;
        for (int j = 0; j < neuron->outgoing_count; j++)
        {
            tConnection *connection = neuron->outgoing_connections[j];
            double contribution = connection->origin->input * connection->weight;
            // printf("Conexión: Input=%.2f, Peso=%.2f, Contribución=%.2f\n", connection->origin->input, connection->weight, contribution);
            neuron->output += contribution;
        }
        neuron->output += neuron->bias;
        // ReLU
        // neuron->output = neuron->output < 0 ? 0 : neuron->output;
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
                    // printf("currentNeuron_input: %.3f   prevNeuron->output: %.3f   peso: %.3f \n", currentNeuron->input, prevNeuron->output);
                }
            }
        }
    }
}

void CalculateOutputError(tLayer *outputLayer, double expected)
{
    for (int i = 0; i < outputLayer->neuron_count; i++)
    {
        outputLayer->neurons[i]->output_error = expected - outputLayer->neurons[i]->output;
        printf("loss: %f\n", outputLayer->neurons[i]->output_error);
    }
}

void Backpropagate(tLayer *currentLayer, float learning_rate)
{
    double max_weight = 10.0;

    for (int i = 0; i < currentLayer->neuron_count; i++)
    {
        tNeuron *neuron = currentLayer->neurons[i];

        if (neuron->outgoing_count > 0)
        {
            neuron->output_error = 0.0;
            for (int j = 0; j < neuron->outgoing_count; j++)
            {
                tConnection *connection = neuron->outgoing_connections[j];
                neuron->output_error += connection->weight * connection->destination->output_error;
            }
        }

        for (int j = 0; j < neuron->outgoing_count; j++)
        {
            tConnection *connection = neuron->outgoing_connections[j];
            double gradient = neuron->output_error * connection->origin->output;
            connection->weight -= learning_rate * gradient;

            if (connection->weight > max_weight)
                connection->weight = max_weight;
            else if (connection->weight < -max_weight)
                connection->weight = -max_weight;
        }
    }
}

void TrainNetwork(tLayer *entryLayer, tLayer *hiddenLayer1, tLayer *hiddenLayer2, tLayer *outputLayer, double *inputs, double *expected_outputs, int data_count, int epochs, float learning_rate)
{
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        double total_loss = 0.0;
        printf("Epoch %d/%d\n", epoch + 1, epochs);

        for (int i = 0; i < data_count; i++)
        {
            SetInput(entryLayer, inputs[i]);
            ForwardPropagation(entryLayer);
            SetLayerInputFromPreviousLayer(entryLayer, hiddenLayer1);
            ForwardPropagation(hiddenLayer1);
            SetLayerInputFromPreviousLayer(hiddenLayer1, hiddenLayer2);
            ForwardPropagation(hiddenLayer2);
            SetLayerInputFromPreviousLayer(hiddenLayer2, outputLayer);
            ForwardPropagation(outputLayer);
            outputLayer->neurons[0]->output = outputLayer->neurons[0]->input;
            CalculateOutputError(outputLayer, expected_outputs[i]);
            total_loss += outputLayer->neurons[0]->output_error * outputLayer->neurons[0]->output_error;

            Backpropagate(outputLayer, learning_rate);
            Backpropagate(hiddenLayer2, learning_rate);
            Backpropagate(hiddenLayer1, learning_rate);
        }

        double avg_loss = total_loss / data_count;
        printf("Pérdida promedio de la época %d: %f\n", epoch + 1, avg_loss);
    }
}
