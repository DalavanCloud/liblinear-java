package liblinear;

import cc.mallet.types.SparseVector;

public class LinearKernel {
	public double dot(double[] w, SparseVector vec){
		return vec.dotProduct(w);
	}
	
	public void add(double[] w, SparseVector vec, double factor){
		vec.addTo(w, factor);
	}
	
	public double snorm(SparseVector vec){
		double res = 0; 
		for(int i = 0; i != vec.numLocations(); i++)
			res += vec.valueAtLocation(i) * vec.valueAtLocation(i);
		
		return res;
	}

}
