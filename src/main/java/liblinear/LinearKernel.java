package liblinear;

import cc.mallet.types.SparseVector;

public class LinearKernel {
	public static double dot(double[] w, SparseVector vec){
		return vec.dotProduct(w);
	}
	
	public static void add(double[] w, SparseVector vec, double factor){
		vec.addTo(w, factor);
	}
	
	public static double snorm(SparseVector vec){
		double res = vec.twoNorm();
		return res * res;
	}

}
