package liblinear;


class L2R_LrFunction implements Function {

    private final double[] C;
    private final double[] z;
    private final double[] D;
    private final Problem  prob;
    private LinearKernel k = new LinearKernel();

    public L2R_LrFunction( Problem prob, double Cp, double Cn ) {
        int i;
        int l = prob.l;
        int[] y = prob.y;

        this.prob = prob;

        z = new double[l];
        D = new double[l];
        C = new double[l];

        for (i = 0; i < l; i++) {
            if (y[i] == 1)
                C[i] = Cp;
            else
                C[i] = Cn;
        }
    }


    private void Xv(double[] v, double[] Xv) {
        for (int i = 0; i < prob.l; i++)
        	Xv[i] = k.dot(v, prob.x.get(i));
    }

    private void XTv(double[] v, double[] XTv) {
        int l = prob.l;

        for (int i = 0; i != XTv.length; i++)
            XTv[i] = 0;
        
        for (int i = 0; i < l; i++) {
        	k.add(XTv, prob.x.get(i), v[i]);
        }
    }


    public double fun(double[] w) {
        int i;
        double f = 0;
        int[] y = prob.y;
        int l = prob.l;

        Xv(w, z);
        for (i = 0; i < l; i++) {
            double yz = y[i] * z[i];
            if (yz >= 0)
                f += C[i] * Math.log(1 + Math.exp(-yz));
            else
                f += C[i] * (-yz + Math.log(1 + Math.exp(yz)));
        }
        f = 2.0 * f;
        for (i = 0; i < w.length; i++)
            f += w[i] * w[i];
        f /= 2.0;

        return (f);
    }

    public void grad(double[] w, double[] g) {
        int i;
        int[] y = prob.y;
        int l = prob.l;
        //int w_size = get_nr_variable();

        for (i = 0; i < l; i++) {
            z[i] = 1 / (1 + Math.exp(-y[i] * z[i]));
            D[i] = z[i] * (1 - z[i]);
            z[i] = C[i] * (z[i] - 1) * y[i];
        }
        XTv(z, g);

        for (i = 0; i < w.length; i++)
            g[i] = w[i] + g[i];
    }

    public void Hv(double[] s, double[] Hs) {
        int i;
        int l = prob.l;
        double[] wa = new double[l];

        Xv(s, wa);
        for (i = 0; i < l; i++)
            wa[i] = C[i] * D[i] * wa[i];

        XTv(wa, Hs);
        for (i = 0; i != Hs.length; i++)
            Hs[i] = s[i] + Hs[i];
    }
}
