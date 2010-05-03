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
	
	public double dot(double[] w, int index, SparseVector vec, int w_size){
		double res = 0;
        for (int j = 0; j != vec.numLocations(); j++) {
            int w_offset = (vec.indexAtLocation(j)) * w_size;
            res += w[w_offset + index] * vec.valueAtLocation(j);
        }
        return res;
	}
	
	public void add(double[] w, int index, SparseVector vec, double factor, int w_size){
		for (int j = 0; j != vec.numLocations(); j++) {
			int w_offset = (vec.indexAtLocation(j)) * w_size;
            w[w_offset + index] += factor * vec.valueAtLocation(j);
		}
	}
}
