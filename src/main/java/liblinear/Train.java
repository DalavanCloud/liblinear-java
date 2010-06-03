package liblinear;

import static liblinear.Linear.atof;
import static liblinear.Linear.atoi;
import gnu.trove.TDoubleArrayList;
import gnu.trove.TIntArrayList;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;
import java.util.zip.GZIPInputStream;

import cc.mallet.types.SparseVector;


public class Train {

    public static void main(String[] args) throws IOException, InvalidInputDataException {
        new Train().run(args);
    }

    private boolean    bias             = false;
    private boolean   cross_validation = false;
    private String    inputFilename;
    private String    modelFilename;
    private int       nr_fold;
    private Parameter param            = null;
    private Problem   prob             = null;

    private void do_cross_validation() {
        

        long start, stop;
        start = System.currentTimeMillis();
        int[] target = Linear.crossValidation(prob, param, nr_fold);
        stop = System.currentTimeMillis();
        System.out.println("time: " + (stop - start) + " ms");
        int tp = 0;
        int tn = 0; 
        int fp = 0; 
        int fn = 0;
        int pos = 0;
        int neg = 0;

        int total_correct = 0;
        for (int i = 0; i < prob.l; i++)
            if (target[i] == prob.y[i]) ++total_correct;
        
        for (int i = 0; i < prob.l; i++){
        	if(prob.y[i] == 1)
        		pos++;
        	else
        		neg++;
        	
        	if(target[i] == prob.y[i]){
        		if(target[i] == 1)
        			tp++;
        		else
        			tn++;
        	}else{
        		if(prob.y[i] == 1)
        			fp++;
        		else
        			fn++;
        	}
        	
        	
        		
        }

        System.out.printf("correct: %d/%d%n", total_correct, prob.y.length);
        System.out.printf("Cross Validation Accuracy = %g%%%n", 100.0 * total_correct / prob.l);
        System.out.printf("Total positive/negative:%d/%d%n", pos, neg);
        System.out.printf("True positive/negative: %d/%d%n", tp, tn);
        System.out.printf("False positive/negative: %d/%d%n", fp, fn);
    }

    private void exit_with_help() {
        System.out.printf("Usage: train [options] training_set_file [model_file]%n"
            + "options:%n" 
            + "-s type : set type of solver (default 1)%n"  
            + "   0 -- L2-regularized logistic regression%n" 
            + "   1 -- L2-regularized L2-loss support vector classification (dual)%n"
            + "   2 -- L2-regularized L2-loss support vector classification (primal)%n"
            + "   3 -- L2-regularized L1-loss support vector classification (dual)%n" 
            + "   4 -- multi-class support vector classification by Crammer and Singer%n"
            + "   5 -- L1-regularized L2-loss support vector classification%n"
            + "   6 -- L1-regularized logistic regression%n"
            + "-c cost : set the parameter C (default 1)%n"
            + "-e epsilon : set tolerance of termination criterion%n"
            + "   -s 0 and 2%n"
            + "       |f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,%n"
            + "       where f is the primal function and pos/neg are # of%n"
            + "       positive/negative data (default 0.01)%n"
            + "   -s 1, 3, and 4%n"
            + "       Dual maximal violation <= eps; similar to libsvm (default 0.1)%n"
            + "   -s 5 and 6%n"
            + "       |f'(w)|_inf <= eps*min(pos,neg)/l*|f'(w0)|_inf,%n"
            + "       where f is the primal function (default 0.01)%n"
            + "-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)%n"
            + "-wi weight: weights adjust the parameter C of different classes (see README for details)%n"
            + "-v n: n-fold cross validation mode%n"
            + "-q : quiet mode (no outputs)%n"
        );
        System.exit(1);
    }


    Problem getProblem() {
        return prob;
    }

    boolean getBias() {
        return bias;
    }

    Parameter getParameter() {
        return param;
    }

    void parse_command_line(String argv[]) {
        int i;

        // eps: see setting below
        param = new Parameter(SolverType.L2R_L2LOSS_SVC_DUAL, 1, Double.POSITIVE_INFINITY);
        // default values
        bias = false;
        cross_validation = false;

        int nr_weight = 0;

        // parse options
        for (i = 0; i < argv.length; i++) {
            if (argv[i].charAt(0) != '-') break;
            if (++i >= argv.length) exit_with_help();
            switch (argv[i - 1].charAt(1)) {
                case 's':
                    param.solverType = SolverType.values()[atoi(argv[i])];
                    break;
                case 'c':
                    param.setC(atof(argv[i]));
                    break;
                case 'e':
                    param.setEps(atof(argv[i]));
                    break;
                case 'B':
                    bias = atof(argv[i]) >= 0;
                    break;
                case 'w':
                    ++nr_weight;
                    {
                        int[] old = param.weightLabel;
                        param.weightLabel = new int[nr_weight];
                        if(old != null)
                        	System.arraycopy(old, 0, param.weightLabel, 0, nr_weight - 1);
                    }

                    {
                        double[] old = param.weight;
                        param.weight = new double[nr_weight];
                        if(old != null)
                        	System.arraycopy(old, 0, param.weight, 0, nr_weight - 1);
                    }

                    param.weightLabel[nr_weight - 1] = atoi(argv[i - 1].substring(2));
                    param.weight[nr_weight - 1] = atof(argv[i]);
                    break;
                case 'v':
                    cross_validation = true;
                    nr_fold = atoi(argv[i]);
                    if (nr_fold < 2) {
                        System.err.print("n-fold cross validation: n must >= 2\n");
                        exit_with_help();
                    }
                    break;
                case 'q':
                    Linear.disableDebugOutput();
                    break;
                default:
                    System.err.println("unknown option");
                    exit_with_help();
            }
        }

        // determine filenames

        if (i >= argv.length) exit_with_help();

        inputFilename = argv[i];

        if (i < argv.length - 1)
            modelFilename = argv[i + 1];
        else {
            int p = argv[i].lastIndexOf('/');
            ++p; // whew...
            modelFilename = argv[i].substring(p) + ".model";
        }

        if (param.eps == Double.POSITIVE_INFINITY) {
            if (param.solverType == SolverType.L2R_LR || param.solverType == SolverType.L2R_L2LOSS_SVC) {
                param.setEps(0.01);
            } else if (param.solverType == SolverType.L2R_L2LOSS_SVC_DUAL || param.solverType == SolverType.L2R_L1LOSS_SVC_DUAL
                || param.solverType == SolverType.MCSVM_CS) {
                param.setEps(0.1);
            } else if (param.solverType == SolverType.L1R_L2LOSS_SVC || param.solverType == SolverType.L1R_LR) {
                param.setEps(0.01);
            }
        }
    }

    /**
     * reads a problem from LibSVM format
     * @param filename the name of the svm file
     * @throws IOException obviously in case of any I/O exception ;)
     * @throws InvalidInputDataException if the input file is not correctly formatted
     */
    public static Problem readProblem(File file, boolean bias) throws IOException, InvalidInputDataException {
    	InputStream is = new FileInputStream(file);
    	if(file.getName().endsWith(".gz"))
    		is = new GZIPInputStream(is);
    	
        BufferedReader fp = new BufferedReader(new InputStreamReader(is));
        List<Integer> vy = new ArrayList<Integer>();
        List<SparseVector> vx = new ArrayList<SparseVector>();
        int max_index = 0;

        int lineNr = 0;

        try {
            while (true) {
                String line = fp.readLine();
                if (line == null) 
                	break;
                lineNr++;
                int commentPos = line.indexOf('#');
                
                if(commentPos != -1)
                	line = line.substring(0, commentPos);
                
                StringTokenizer st = new StringTokenizer(line, " \t\n\r\f:");
                
                if(!st.hasMoreTokens())
                	continue;
                
                String token = st.nextToken();
                
                if(lineNr % 10000 == 0){
                	System.out.printf("%d..", lineNr);
                	System.out.flush();
                }

                try {
                    vy.add(atoi(token));
                } catch (NumberFormatException e) {
                    throw new InvalidInputDataException("invalid label: " + token, file, lineNr, e);
                }

                int m = st.countTokens() / 2;
                int indexBefore = 0;
                TIntArrayList indexes = new TIntArrayList();
                TDoubleArrayList values = new TDoubleArrayList();
                for (int j = 0; j < m; j++) {

                    token = st.nextToken();
                    int index;
                    try {
                        index = atoi(token);
                    } catch (NumberFormatException e) {
                        throw new InvalidInputDataException("invalid index: " + token, file, lineNr, e);
                    }

                    // assert that indices are valid and sorted
                    if (index < 0) throw new InvalidInputDataException("invalid index: " + index, file, lineNr);
                    if (index <= indexBefore) throw new InvalidInputDataException("indices must be sorted in ascending order", file, lineNr);
                    indexBefore = index;

                    token = st.nextToken();
                    try {
                        double value = atof(token);
                        indexes.add(index);
                        values.add(value);
                        max_index = Math.max(max_index, index);
                        
                    } catch (NumberFormatException e) {
                        throw new InvalidInputDataException("invalid value: " + token, file, lineNr);
                    }
                }
                if(bias){
                	indexes.add(0);
                	values.add(1.0);
                }

                vx.add(new SparseVector(indexes.toNativeArray(), values.toNativeArray(), false, false));
            }
            System.out.printf("%d. Done!%n", lineNr);
            return constructProblem(vy, vx, max_index, bias);
        }
        finally {
            fp.close();
            is.close();
        }
    }

    void readProblem(String filename) throws IOException, InvalidInputDataException {
        prob = Train.readProblem(new File(filename), bias);
    }

    private static Problem constructProblem(List<Integer> vy, List<SparseVector> vx, int max_index, boolean bias) {
        Problem prob = new Problem();
        prob.bias = bias;
        prob.l = vy.size();
        prob.n = max_index + 1;
        if (bias) {
            prob.n++;
        }
        prob.x = vx;

        for (int i = 0; i < prob.l; i++) {
        	SparseVector x = vx.get(i);
        	int x_size = x.numLocations();
        	if (bias){
        		assert x.indexAtLocation(x_size - 1) == 0;
        		x.getIndices()[x_size - 1] = max_index + 1;
        	}else{
        		assert x.indexAtLocation(x_size - 1) != 0;
        	}
        }

        prob.y = new int[prob.l];
        for (int i = 0; i < prob.l; i++)
            prob.y[i] = vy.get(i);

        return prob;
    }

    private void run(String[] args) throws IOException, InvalidInputDataException {
        parse_command_line(args);
        readProblem(inputFilename);
        if (cross_validation)
            do_cross_validation();
        else {
            Model model = Linear.train(prob, param);
            Linear.saveModel(new File(modelFilename), model);
        }
    }
}
