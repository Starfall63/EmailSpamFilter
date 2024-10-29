import numpy as np
from IPython.display import HTML,Javascript, display


def main():
    training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",").astype(int)

    classifier = create_classifier()

    testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(int)
    test_data = testing_spam[:, 1:]
    test_labels = testing_spam[:, 0]

    predictions = classifier.predict(test_data)
    accuracy = np.count_nonzero(predictions == test_labels)/test_labels.shape[0]
    print(f"Accuracy on test data is: {accuracy}")




class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
          self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
          self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
          #forward pass of the input data
          self.output = np.dot(np.array(inputs), self.weights) + self.biases

#Activation Rectified Linear Unit
class Activation_ReLU():
      def forward(self,inputs):
            self.output = np.maximum(0, inputs)


class Activation_Softmax:
      def forward(self, inputs):
            exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) #input = max(inputs) to prevent overflow
            probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
            self.output = probabilities


class Loss:
      def calculate(self, output, y):
            sample_losses = self.forward(output, y)
            data_loss = np.mean(sample_losses)
            return data_loss

#Categorical Cross Entropy used for loss function
class Loss_CategoricalCrossEntropy(Loss):
      def forward(self, y_pred, y_true):
            samples = len(y_pred)
            y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

            if len(y_true.shape) == 1: #scalar values 
                  correct_confidences = y_pred_clipped[range(samples), y_true]
            elif len(y_true.shape) == 2: #one-hot encoded vectors
                  correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
            
            negative_log_likelihoods = -np.log(correct_confidences)
            return negative_log_likelihoods


class SpamClassifier:
    def __init__(self, k):
        self.k = k
        
    def train(self):
            '''
            emailData = training_spam[:,1:]
            targets = training_spam[:,0]

            layer1 = Layer_Dense(54, 4)
            activation1 = Activation_ReLU()
            layer2 = Layer_Dense(4, 2)
            activation2 = Activation_Softmax()

            layer1.forward(emailData)
            activation1.forward(layer1.output)
            layer2.forward(activation1.output)
            activation2.forward(layer2.output)

            loss_function = Loss_CategoricalCrossEntropy()

            lowest_loss = 9999999
            self.best_layer1_weights = layer1.weights.copy()
            self.best_layer1_biases = layer1.biases.copy()
            self.best_layer2_weights = layer2.weights.copy()
            self.best_layer2_biases = layer2.biases.copy()

            for iteration in range(500000):
                  layer1.weights += 0.05 * np.random.randn(54, 4)
                  layer1.biases += 0.05 * np.random.randn(1, 4)
                  layer2.weights += 0.05 * np.random.randn(4, 2)
                  layer2.biases += 0.05 * np.random.randn(1, 2)

                  layer1.forward(emailData)
                  activation1.forward(layer1.output)
                  layer2.forward(activation1.output)
                  activation2.forward(layer2.output)

                  loss = loss_function.calculate(activation2.output, targets)

                  predictions = np.argmax(activation2.output, axis=1)
                  accuracy = np.mean(predictions == targets)

                  if loss < lowest_loss:
                        self.best_layer1_weights = layer1.weights.copy()
                        self.best_layer1_biases = layer1.biases.copy()
                        self.best_layer2_weights = layer2.weights.copy()
                        self.best_layer2_biases = layer2.biases.copy()
                        lowest_loss = loss
                  else:
                        layer1.weights = self.best_layer1_weights.copy()
                        layer1.biases = self.best_layer1_biases.copy()
                        layer2.weights = self.best_layer2_weights.copy()
                        layer2.biases = self.best_layer2_biases.copy()
                        '''
            pass

        
    def predict(self, data):
        layer1 = Layer_Dense(54, 4)
        activation1 = Activation_ReLU()
        layer2 = Layer_Dense(4, 2)
        activation2 = Activation_Softmax()
        '''
        layer1.weights = self.best_layer1_weights
        layer1.biases = self.best_layer1_biases
        layer2.weights = self.best_layer2_weights
        layer2.biases = self.best_layer2_biases
        '''
        
        layer1.weights = np.array([[-0.54010093, -0.00560699, -0.31908303,  0.6843123 ],
 [-0.7380759 ,  0.27247742, -1.7358068 ,  0.60603046],
 [-1.1063097 ,  0.10985462,  1.7958045 , -2.2957914 ],
 [-0.29899642,  0.745205  ,  0.04416705, -0.23880857],
 [ 0.8237654 , -1.0009528 , -0.99239236, -1.0768088 ],
 [-0.2997505 ,  2.446705  ,  2.248597  ,  0.41405016],
 [-0.74659145,  1.3046942 , -0.5293365 , -0.23831338],
 [-0.48541775,  0.8804031 , -0.5881959 ,  1.3618095 ],
 [ 0.817464  ,  0.8887871 ,  0.5571179 ,  0.28964993],
 [ 0.8327633 , -1.3966434 , -2.0829163 , -0.7878155 ],
 [ 1.1636223 ,  0.13593023,  0.8996365 , -0.8043828 ],
 [-0.348454  ,  0.39279616,  1.3410344 , -0.11958574],
 [-2.2912538 ,  0.12952295,  1.9174719 , -0.7329556 ],
 [ 0.43108547, -0.888233  ,  0.08036306, -0.15299161],
 [-0.07657088,  0.5995495 , -0.05198853,  0.32671317],
 [-0.33212948,  0.9506355 , -0.5288691 ,  0.7817274 ],
 [ 0.47441834,  1.1937647 ,  1.5508822 , -0.89465255],
 [-0.84072495,  1.098115  ,  0.14615148,  2.4286668 ],
 [-1.5751227 ,  0.28595594, -1.2300663 ,  1.0853405 ],
 [-0.24450588,  0.6858476 , -1.0048852 , -0.96236694],
 [ 0.52456033,  1.7300999 ,  0.07441908,  1.3167669 ],
 [-0.35008258, -0.0774644 , -1.157458  , -0.05268058],
 [-0.2826965 ,  2.183274  ,  1.2800673 ,  0.6994735 ],
 [ 0.08367345,  1.5314219 ,  1.4603984 , -1.4581187 ],
 [-0.53541327, -1.2540592 ,  2.3681927 , -1.6156273 ],
 [-0.74553704,  0.25112867,  0.7523132 , -1.8161014 ],
 [-0.3848562 , -2.2614799 ,  2.88642   ,  1.4580674 ],
 [ 1.6863359 ,  0.50006247,  1.4857255 ,  0.03065994],
 [ 2.1011412 , -0.95298666, -0.9079003 ,  1.3653605 ],
 [ 0.36655787, -0.942094  , -1.0267309 ,  0.6378301 ],
 [-1.0060166 ,  0.29339135,  0.41893545, -0.09407391],
 [-0.89600366,  1.3267232 , -0.01150009, -0.11601385],
 [-0.3666477 ,  0.48367733,  0.66362596, -0.34223655],
 [-2.2724547 , -0.8493859 ,  1.775492  ,  0.3366698 ],
 [ 1.3899391 , -0.21438132,  1.5893358 , -0.7439535 ],
 [-0.70512784,  0.4275639 , -0.14500608, -1.3239747 ],
 [-0.28518814,  1.2086154 ,  0.8659204 ,  0.8855314 ],
 [-0.93683773,  2.34095   ,  1.7525328 , -0.26258314],
 [ 2.0832448 ,  0.07558502, -0.2750589 ,  1.4037901 ],
 [-0.5907597 , -1.2751726 , -1.3879913 ,  0.19971056],
 [-0.5006639 , -0.9417657 ,  0.32501   ,  1.1219809 ],
 [-0.14681481, -1.5477964 ,  3.0128107 ,  0.05357721],
 [-0.31016487,  0.03041348, -1.2392248 ,  0.92052484],
 [ 2.0886676 , -0.6366212 ,  1.0464267 ,  0.8894521 ],
 [ 0.8461748 , -0.07893826,  1.2995256 ,  0.03543937],
 [ 1.9654197 , -4.2052197 ,  0.22423182,  0.615933  ],
 [ 0.47200847,  1.0406494 ,  1.7654041 , -0.40165654],
 [-1.1554393 , -0.49612185,  0.31785053, -1.2282637 ],
 [-0.82042354,  0.0958738 , -0.7826756 ,  0.4254088 ],
 [ 0.06835449, -0.27633166,  0.59448135, -0.8951759 ],
 [ 1.4178667 ,  0.7091023 ,  0.70074856,  0.89944905],
 [ 0.09393227,  2.2310433 ,  0.77769303,  1.105141  ],
 [ 1.4300119 ,  0.78895134,  0.3513684 ,  0.39869514],
 [ 0.9187338 , -0.4296623 ,  0.52149373,  0.2040069 ]])

        layer1.biases = np.array([[-0.54528934,  0.3184455 ,  0.52240336,  0.00543118]])
        layer2.weights = np.array([[-1.403081 ,  1.7584153],
 [-2.130975 ,  1.3649839],
 [ 1.8843361, -1.0144899],
 [ 0.8484387, -2.558999 ]])
        layer2.biases = np.array([[-0.06644698, -1.8758985 ]])


        layer1.forward(data)
        activation1.forward(layer1.output)
        layer2.forward(activation1.output)
        activation2.forward(layer2.output)

        class_predictions = np.argmax(activation2.output, axis=1)

        
        return class_predictions
    

def create_classifier():
    classifier = SpamClassifier(k=1)
    classifier.train()
    return classifier



if __name__ == '__main__':
    main()
    



