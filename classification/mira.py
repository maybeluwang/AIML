# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.

        ## Something you might need to use:
        ## trainingData[i]: a feature vector, an util.Counter()
        ## trainingLabels[i]: label for each trainingData[i]
        ## self.weights[label]: weight vector for a label (class), an util.Counter()
        ## self.classify(data): this method might be needed in validation
        ## Cgrid: a list of constant C
        """
        "*** YOUR CODE HERE ***"
        for iteration in range(self.max_iterations):
            print "Starting MIRA iteration ", iteration, "..."
            Score_C = []
            Weights_C = {}
            OriginWeights = self.weights.copy()
            for const in Cgrid:
                for FeatureVector, TrueLabel in zip(trainingData, trainingLabels):
                    score = util.Counter()
                    for label in self.legalLabels:
                        score[label] = self.weights[label]*FeatureVector
                    PredLabel = score.argMax()

                    Tau = ((self.weights[PredLabel]-self.weights[TrueLabel])*FeatureVector+1.0)/2.0/(FeatureVector*FeatureVector)
                    Tau = min([const, Tau])
                    delta = FeatureVector.copy()
                    for key, value in delta.items():
                        delta[key] = value * Tau
                    if PredLabel != TrueLabel:
                        self.weights[TrueLabel] += delta
                        self.weights[PredLabel] -= delta
            
                Weights_C[const] = self.weights
                Score_C.append(sum(int(y_true==y_pred) for y_true, y_pred in zip(validationLabels, self.classify(validationData))))
                self.weights = OriginWeights.copy()

            BestConst, BestValScore = Cgrid[0], -1
            for const, ValScore in zip(Cgrid, Score_C):
                if ValScore > BestValScore:
                    BestConst, BestValScore = const, ValScore
                elif ValScore == BestValScore:
                    if const < BestConst:
                        BestConst, BestValScore = const, ValScore

            self.weights = Weights_C[BestConst]

    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


