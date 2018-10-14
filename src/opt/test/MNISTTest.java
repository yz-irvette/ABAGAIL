package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying MNIST images as digits 0-9. 
 *
 * Branched from AbaloneTest.java
 */
public class MNISTTest {
	// CHANGE_ME: should the absolute path of your project location + "src/opt/test/" if using eclipse
	private static String filepath = "src/opt/test/";  // "F:/Workspace/RandomizedOptimization/src/opt/test/";
    private static Instance[] instances = initializeInstances(filepath + "mnist_train_images",
    		                                                  filepath + "mnist_train_labels");
    private static Instance[] instancesTest = initializeInstances(filepath + "mnist_test_images",
                                                                  filepath + "mnist_test_labels");

    private static int inputLayer = 28*28, hiddenLayer = 28, outputLayer = 10, trainingIterations = 1000;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        for(int i = 0; i < oa.length; i++) {
            System.out.println("Build network with optimization approach " + oaNames[i]);
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        for(int i = 0; i < oa.length; i++) {
            System.out.println("\n--------------------------- " + oaNames[i] + " ---------------------------");
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0, correct2 = 0, incorrect2 = 0;
            train(oa[i], networks[i], oaNames[i]); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            System.out.println("\nTraining done. Start testing...");
            start = System.nanoTime();
            for(int j = 0; j < instancesTest.length; j++) {
                networks[i].setInputValues(instancesTest[j].getData());
                networks[i].run();
                for (int k = 0; k < 10; k++) {
                	if (instancesTest[j].getLabel().getDiscrete(k) == 1) {
                		if (networks[i].getOutputLayer().getGreatestActivationIndex() == k) {
                			correct2++;
                		} else {
                			incorrect2++;
                		}
                		break;
                	}
                }
                for (int k = 0; k < 10; k++) {
                	if (Math.abs(instancesTest[j].getLabel().getContinuous(k) - networks[i].getOutputValues().get(k)) <= 0.5) {
                		correct++;
	                } else {
	                	incorrect++;
	                }
                }
            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            results +=  "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct/10.0 + ", " + correct2 + " instances." +
                        "\nIncorrectly classified " + incorrect/10.0 + ", " + incorrect2 + " instances.\nPercent correctly classified: "
                        + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
            System.out.println(results);
        }
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }

            if (i % 20 == 0) {
            	System.out.println("Round " + i + ": " + df.format(error));
            }
        }
    }

    // From https://github.com/kpchand/mnist-neuralnetwork/blob/master/src/Loader.java
    private static int readInt(InputStream in) throws IOException{
        // Data is stored in high endian format so make it low endian.
        int d;
        int[] b = new int[4];
        for(int i = 0; i < 4; i++)
            b[i] = in.read();
        d = b[3] | b[2] << 8 | b[1] << 16 | b[0] << 24;
        return d;
    }

    private static Instance[] initializeInstances(String imagesPath, String labelsPath) {
        try {
            InputStream images = new FileInputStream(imagesPath);
            readInt(images);
            int num = readInt(images);
            int rows = readInt(images);
            int cols = readInt(images);
            int size = rows * cols;
            double[][][] attributes = new double[num][][];
            System.out.println("Num: " + num);
            for(int i = 0; i < attributes.length; i++) {
                attributes[i] = new double[2][];
                attributes[i][0] = new double[size];
                for(int j = 0; j < size; j++) {
                    attributes[i][0][j] = images.read() / 255.0;
                }
            }
            
            InputStream labels = new FileInputStream(labelsPath);
            readInt(labels);
            num = readInt(labels);
            System.out.println("Num: " + num);
            for(int i = 0; i < attributes.length; i++) {
                attributes[i][1] = new double[10];
                int label = labels.read();
                attributes[i][1][label] = 1;
            }

            Instance[] instances = new Instance[attributes.length];
            for(int i = 0; i < instances.length; i++) {
                instances[i] = new Instance(attributes[i][0]);
                // classifications range from 0 to 30; split into 0 - 14 and 15 - 30
                instances[i].setLabel(new Instance(attributes[i][1]));
            }
            return instances;
        }
        catch(Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}
