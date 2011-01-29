package org.apache.mahout.math.ssvd;

import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.EigenDecompositionImpl;
import org.apache.commons.math.linear.RealMatrix;

/**
 * wraps appropriate eigen solver for BBt matrix. 
 * Can be either colt or apache commons math. <P>
 * 
 * At the moment it is apache commons math which 
 * is only in mahout-math dependencies. <P>
 * 
 * I will be happy to switch this to Colt eigensolver 
 * if it is proven reliable (i experience internal errors 
 * and unsorted singular values at some point).
 * 
 * But for now commons-math seems to be more reliable.
 * 
 * 
 * @author Dmitriy
 *
 */
public class EigenSolverWrapper {
    
    private double[]    m_eigenvalues;
    private double[][]  m_uHat;

    public EigenSolverWrapper(double[][] bbt ) {
        super();
        int dim=bbt.length;
        EigenDecompositionImpl evd2=new EigenDecompositionImpl(new Array2DRowRealMatrix(bbt),0);
        m_eigenvalues=evd2.getRealEigenvalues();
        RealMatrix uHatrm= evd2.getV();
        m_uHat = new double[dim][];
        for ( int i = 0; i < dim; i++ ) m_uHat [i]=uHatrm.getRow(i);
    }
    
    public double[][] getUHat() { return m_uHat; }
    public double[] getEigenValues() { return m_eigenvalues; } 
    
    

    
}
