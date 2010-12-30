/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.math.hadoop.stochasticsvd;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.apache.mahout.math.AbstractVector;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

/**
 * Givens Thin solver. Standard Givens operations are reordered in a way 
 * that helps us to push them thru MapReduce operations in a block fashion. 
 * 
 * @author Dmitriy
 *
 */
public class GivensThinSolver {

    private static final double s_epsilon = 1e-10;

    // private double[][] m_rTilde;
    private Vector m_aRowV;
    private double[] m_aRow, m_qtRow;
    private UpperTriangular m_rTilde;
    private TriangularRowView m_rTildeRowView, m_rTildeRowView2;
    private double[][] m_qt;
    private int m_qtStartRow;
    private int m_m, m_n; // m-row cnt, n- column count, m>=n
    private int m_cnt;
    private double[] m_cs = new double[2];

    public GivensThinSolver(int m, int n) {
        super();

        if (!(m >= n))
            throw new IllegalArgumentException(
                    "Givens thin QR: must be true: m>=n");

        m_m = m;
        m_n = n;
        m_rTilde = new UpperTriangular(n);
        m_rTildeRowView = new TriangularRowView(m_rTilde);
        m_rTildeRowView2 = new TriangularRowView(m_rTilde);
        m_qt = new double[n][];
        m_aRowV = new DenseVector(m_aRow = new double[n], true);
        m_qtRow = new double[m];

        for (int i = 0; i < n; i++) {
            m_qt[i] = new double[m_m];
        }
        m_cnt = 0;
    }
    
    public void reset ( ) { 
        m_cnt =0 ;
    }

    public void solve(Matrix a) {

        assert a.rowSize() == m_m;
        assert a.columnSize() == m_n;

        double[] aRow = new double[m_n];
        for (int i = 0; i < m_m; i++) {
            Vector aRowV = a.getRow(i);
            for (int j = 0; j < m_n; j++)
                aRow[j] = aRowV.getQuick(j);
            appendRow(aRow);
        }
    }

    public boolean isFull() {
        return m_cnt == m_m;
    }

    public int getM() {
        return m_m;
    }

    public int getN() {
        return m_n;
    }

    public int getCnt() {
        return m_cnt;
    }

    public void adjust(int newM) {
        if (newM == m_m)
            return; // no adjustment is required.
        if (newM < m_n)
            throw new IllegalArgumentException("new m can't be less than n");
        if (newM < m_cnt)
            throw new IllegalArgumentException(
                    "new m can't be less than rows accumulated");
        m_qtRow = new double[newM];

        // grow or shrink qt rows
        if (newM > m_m) {
            // grow qt rows
            for (int i = 0; i < m_n; i++) {
                m_qt[i] = Arrays.copyOf(m_qt[i], newM);
                System.arraycopy(m_qt[i], 0, m_qt[i], newM - m_m, m_m);
                Arrays.fill(m_qt[i], 0, newM - m_m, 0);
            }
        } else {
            // shrink qt rows
            for (int i = 0; i < m_n; i++) {
                m_qt[i] = Arrays.copyOfRange(m_qt[i], m_m - newM, m_m);
            }
        }

        m_m = newM;

    }

    public void trim() {
        adjust(m_cnt);
    }

    /**
     * api for row-by-row addition
     * 
     * @param aRow
     */
    public void appendRow(double[] aRow) {
        if (m_cnt >= m_m)
            throw new RuntimeException(
                    "thin QR solver fed more rows than initialized for");
        try {
            // moving pointers around is inefficient but
            // for the sanity's sake i am keeping it this way so i don't
            // have to guess how R-tilde index maps to actual block index

            Arrays.fill(m_qtRow, 0);
            m_qtRow[m_m - m_cnt - 1] = 1;
            int height = m_cnt > m_n ? m_n : m_cnt;
            System.arraycopy(aRow, 0, m_aRow, 0, m_n);

            if (height > 0) {
                givens(m_aRow[0], m_rTilde.getQuick(0, 0), m_cs);
                applyGivensInPlace(m_cs[0], m_cs[1], m_aRowV,
                        m_rTildeRowView.setViewedRow(0), 0, m_n);
                applyGivensInPlace(m_cs[0], m_cs[1], m_qtRow, _getQtRow(0), 0, m_m);
            }

            for (int i = 1; i < height; i++) {
                givens(m_rTilde.getQuick(i - 1, i), m_rTilde.getQuick(i, i),
                        m_cs);
                // givens(m_rTilde[i][i], m_rTilde[i+1][i], m_cs);
                applyGivensInPlace(m_cs[0], m_cs[1],
                        m_rTildeRowView2.setViewedRow(i - 1),
                        m_rTildeRowView.setViewedRow(i), i, m_n - i);
                applyGivensInPlace(m_cs[0], m_cs[1], _getQtRow(i - 1), _getQtRow(i), 0,
                        m_m);
            }
            // push qt and r-tilde 1 row down
            // just sqp the references to reduce GC churning
            _pushQtDown();
            double[] swap=_getQtRow(0);
            _setQtRow(0,m_qtRow);
            m_qtRow=swap;

            // triangular push -- obviously, less efficient than
            // just reference swap above -- but saves us some memory, cpu time
            // is cheap
            for (int i = m_n - 1; i > 0; i--) {
                // copy (i-1)th row into i-th row ignoring main diagonal item
                // which must be 0 now
                assert m_rTilde.getQuick(i - 1, i - 1) <= s_epsilon;
                for (int j = i; j < m_n; j++)
                    m_rTilde.setQuick(i, j, m_rTilde.getQuick(i - 1, j));
            }
            for (int i = 0; i < m_n; i++)
                m_rTilde.setQuick(0, i, m_aRow[i]);

        } finally {
            m_cnt++;
        }
    }
    
    private double[] _getQtRow ( int row ) {
        
        return m_qt[(row+=m_qtStartRow)>=m_n?row-m_n:row];
    }
    private void _setQtRow ( int row, double[] qtRow ) { 
        m_qt[(row+=m_qtStartRow)>=m_n?row-m_n:row]=qtRow;
    }
    private void _pushQtDown () {
        m_qtStartRow=m_qtStartRow==0?m_n-1:m_qtStartRow-1;
    }

    // warning: both of these return actually n+1 rows with the last one being
    // not interesting.
    public UpperTriangular getRTilde() {
        return m_rTilde;
    }

    public double[][] getThinQtTilde() {
        if ( m_qtStartRow!=0 ) { 
            // rotate qt rows into place
            double[][] qt=new double[m_n][]; // double[~500][], once per block, not a big deal.
            System.arraycopy(m_qt, m_qtStartRow, qt, 0, m_n-m_qtStartRow);
            System.arraycopy(m_qt, 0, qt,m_n-m_qtStartRow,m_qtStartRow);
            return qt;
        }
        return m_qt;
    }

    public static void applyGivensInPlace(double c, double s, double[] row1,
            double[] row2, int offset, int len) {

        int n = offset + len;
        for (int j = offset; j < n; j++) {
            double tau1 = row1[j];
            double tau2 = row2[j];
            row1[j] = c * tau1 - s * tau2;
            row2[j] = s * tau1 + c * tau2;
        }
    }

    public static void applyGivensInPlace(double c, double s, Vector row1,
            Vector row2, int offset, int len) {

        int n = offset + len;
        for (int j = offset; j < n; j++) {
            double tau1 = row1.getQuick(j);
            double tau2 = row2.getQuick(j);
            row1.setQuick(j, c * tau1 - s * tau2);
            row2.setQuick(j, s * tau1 + c * tau2);
        }
    }

    public static void applyGivensInPlace(double c, double s, int i, int k,
            Matrix mx) {
        int n = mx.columnSize();

        for (int j = 0; j < n; j++) {
            double tau1 = mx.get(i, j);
            double tau2 = mx.get(k, j);
            mx.set(i, j, c * tau1 - s * tau2);
            mx.set(k, j, s * tau1 + c * tau2);
        }
    }

    public static void fromRho(double rho, double[] csOut) {
        if (rho == 1) {
            csOut[0] = 0;
            csOut[1] = 1;
            return;
        }
        if (Math.abs(rho) < 1) {
            csOut[1] = 2 * rho;
            csOut[0] = Math.sqrt(1 - csOut[1] * csOut[1]);
            return;
        }
        csOut[0] = 2 / rho;
        csOut[1] = Math.sqrt(1 - csOut[0] * csOut[0]);
    }

    public static void givens(double a, double b, double[] csOut) {
        if (b == 0) {
            csOut[0] = 1;
            csOut[1] = 0;
            return;
        }
        if (Math.abs(b) > Math.abs(a)) {
            double tau = -a / b;
            csOut[1] = 1 / Math.sqrt(1 + tau * tau);
            csOut[0] = csOut[1] * tau;
        } else {
            double tau = -b / a;
            csOut[0] = 1 / Math.sqrt(1 + tau * tau);
            csOut[1] = csOut[0] * tau;
        }
    }

    public static double toRho(double c, double s) {
        if (c == 0)
            return 1;
        if (Math.abs(s) < Math.abs(c))
            return Math.signum(c) * s / 2;
        else
            return Math.signum(s) * 2 / c;
    }

    public static void mergeR(UpperTriangular r1, UpperTriangular r2) {
        TriangularRowView r1Row = new TriangularRowView(r1), r2Row = new TriangularRowView(
                r2);
        int kp = r1Row.size();
        assert kp == r2Row.size();

        double[] cs = new double[2];

        for (int v = 0; v < kp; v++) {
            for (int u = v; u < kp; u++) {
                givens(r1Row.setViewedRow(u).get(u), r2Row.setViewedRow(u - v)
                        .get(u), cs);
                applyGivensInPlace(cs[0], cs[1], r1Row, r2Row, u, kp - u);
            }
        }
    }

    public static void mergeR(double[][] r1, double[][] r2) {
        int kp = r1[0].length;
        assert kp == r2[0].length;

        double[] cs = new double[2];

        for (int v = 0; v < kp; v++) {
            for (int u = v; u < kp; u++) {
                givens(r1[u][u], r2[u - v][u], cs);
                applyGivensInPlace(cs[0], cs[1], r1[u], r2[u - v], u, kp - u);
            }
        }

    }

    public static void mergeRonQ(UpperTriangular r1, UpperTriangular r2,
            double[][] qt1, double[][] qt2) {
        TriangularRowView r1Row = new TriangularRowView(r1), r2Row = new TriangularRowView(
                r2);
        int kp = r1Row.size();
        assert kp == r2Row.size();
        assert kp == qt1.length;
        assert kp == qt2.length;

        int r = qt1[0].length;
        assert qt2[0].length == r;

        double[] cs = new double[2];

        for (int v = 0; v < kp; v++) {
            for (int u = v; u < kp; u++) {
                givens(r1Row.setViewedRow(u).get(u), r2Row.setViewedRow(u - v)
                        .get(u), cs);
                applyGivensInPlace(cs[0], cs[1], r1Row, r2Row, u, kp - u);
                applyGivensInPlace(cs[0], cs[1], qt1[u], qt2[u - v], 0, r);
            }
        }
    }

    public static void mergeRonQ(double[][] r1, double[][] r2, double[][] qt1,
            double[][] qt2) {

        int kp = r1[0].length;
        assert kp == r2[0].length;
        assert kp == qt1.length;
        assert kp == qt2.length;

        int r = qt1[0].length;
        assert qt2[0].length == r;
        double[] cs = new double[2];

        // pairwise givens(a,b) so that a come off main diagonal in r1
        // and bs come off u-th upper subdiagonal in r2.
        for (int v = 0; v < kp; v++) {
            for (int u = v; u < kp; u++) {
                givens(r1[u][u], r2[u - v][u], cs);
                applyGivensInPlace(cs[0], cs[1], r1[u], r2[u - v], u, kp - u);
                applyGivensInPlace(cs[0], cs[1], qt1[u], qt2[u - v], 0, r);
            }
        }
    }

    // returns merged Q (which in this case is the qt1)
    public static double[][] mergeQrUp(double[][] qt1, double[][] r1,
            double[][] r2) {
        int kp = qt1.length;
        int r = qt1[0].length;

        double[][] qTilde = new double[kp][];
        for (int i = 0; i < kp; i++)
            qTilde[i] = new double[r];
        mergeRonQ(r1, r2, qt1, qTilde);
        return qt1;
    }

    // returns merged Q (which in this case is the qt1)
    public static double[][] mergeQrUp(double[][] qt1, UpperTriangular r1,
            UpperTriangular r2) {
        int kp = qt1.length;
        int r = qt1[0].length;

        double[][] qTilde = new double[kp][];
        for (int i = 0; i < kp; i++)
            qTilde[i] = new double[r];
        mergeRonQ(r1, r2, qt1, qTilde);
        return qt1;
    }

    public static double[][] mergeQrDown(double[][] r1, double[][] qt2,
            double[][] r2) {
        int kp = qt2.length;
        int r = qt2[0].length;

        double[][] qTilde = new double[kp][];
        for (int i = 0; i < kp; i++)
            qTilde[i] = new double[r];
        mergeRonQ(r1, r2, qTilde, qt2);
        return qTilde;

    }

    public static double[][] mergeQrDown(UpperTriangular r1, double[][] qt2,
            UpperTriangular r2) {
        int kp = qt2.length;
        int r = qt2[0].length;

        double[][] qTilde = new double[kp][];
        for (int i = 0; i < kp; i++)
            qTilde[i] = new double[r];
        mergeRonQ(r1, r2, qTilde, qt2);
        return qTilde;

    }

    public static double[][] computeQtHat(double[][] qt, int i,
            Iterator<UpperTriangular> rIter) {
        UpperTriangular rTilde = rIter.next();
        for (int j = 1; j < i; j++)
            mergeR(rTilde, rIter.next());
        if (i > 0)
            qt = mergeQrDown(rTilde, qt, rIter.next());
        for (int j = i + 1; rIter.hasNext(); j++)
            qt = mergeQrUp(qt, rTilde, rIter.next());
        return qt;
    }

    // test helpers
    public static boolean isOrthonormal(double[][] qt,
            boolean insufficientRank, double epsilon) {
        int n = qt.length;
        int rank = 0;
        for (int i = 0; i < n; i++) {
            Vector e_i = new DenseVector(qt[i], true);

            double norm = e_i.norm(2);

            if (Math.abs(1 - norm) < epsilon)
                rank++;
            else if (Math.abs(norm) > epsilon)
                return false; // not a rank deficiency, either

            for (int j = 0; j <= i; j++) {
                Vector e_j = new DenseVector(qt[j], true);
                double dot = e_i.dot(e_j);
                if (!(Math.abs((i == j && rank > j ? 1 : 0) - dot) < epsilon))
                    return false;
            }
        }
        return (!insufficientRank && rank == n)
                || (insufficientRank && rank < n);

    }

    public static boolean isOrthonormalBlocked(Iterable<double[][]> qtHats,
            boolean insufficientRank, double epsilon) {
        int n = qtHats.iterator().next().length;
        int rank = 0;
        for (int i = 0; i < n; i++) {
            List<Vector> e_i = new ArrayList<Vector>();
            // Vector e_i=new DenseVector (qt[i],true);
            for (double[][] qtHat : qtHats)
                e_i.add(new DenseVector(qtHat[i], true));

            double norm = 0;
            for (Vector v : e_i)
                norm += v.dot(v);
            norm = Math.sqrt(norm);
            if (Math.abs(1 - norm) < epsilon)
                rank++;
            else if (Math.abs(norm) > epsilon)
                return false; // not a rank deficiency, either

            for (int j = 0; j <= i; j++) {
                List<Vector> e_j = new ArrayList<Vector>();
                for (double[][] qtHat : qtHats)
                    e_j.add(new DenseVector(qtHat[j], true));

                // Vector e_j = new DenseVector ( qt[j], true);
                double dot = 0;
                for (int k = 0; k < e_i.size(); k++)
                    dot += e_i.get(k).dot(e_j.get(k));
                if (!(Math.abs((i == j && rank > j ? 1 : 0) - dot) < epsilon))
                    return false;
            }
        }
        return (!insufficientRank && rank == n)
                || (insufficientRank && rank < n);

    }

    private static class TriangularRowView extends AbstractVector {
        private UpperTriangular m_viewed;
        private int m_rowNum;

        public TriangularRowView(UpperTriangular viewed) {
            super(viewed.columnSize());
            m_viewed = viewed;

        }

        TriangularRowView setViewedRow(int row) {
            m_rowNum = row;
            return this;
        }

        @Override
        public boolean isDense() {
            return true;
        }

        @Override
        public boolean isSequentialAccess() {
            return false;
        }

        @Override
        public Iterator<Element> iterator() {
            throw new UnsupportedOperationException();
        }

        @Override
        public Iterator<Element> iterateNonZero() {
            throw new UnsupportedOperationException();
        }

        @Override
        public double getQuick(int index) {
            return m_viewed.getQuick(m_rowNum, index);
        }

        @Override
        public Vector like() {
            throw new UnsupportedOperationException();
        }

        @Override
        public void setQuick(int index, double value) {
            m_viewed.setQuick(m_rowNum, index, value);

        }

        @Override
        public int getNumNondefaultElements() {
            throw new UnsupportedOperationException();
        }

        @Override
        protected Matrix matrixLike(int rows, int columns) {
            throw new UnsupportedOperationException();
        }

    }

    public static class DeepCopyUTIterator implements Iterator<UpperTriangular> {

        private Iterator<UpperTriangular> delegate;

        public DeepCopyUTIterator(Iterator<UpperTriangular> del) {
            super();
            delegate = del;
        }

        public boolean hasNext() {
            return delegate.hasNext();
        }

        public UpperTriangular next() {

            return new UpperTriangular(delegate.next());
        }

        public void remove() {
            delegate.remove();
        }

    }

}
