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

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.EigenDecompositionImpl;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.SingularValueDecomposition;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.UnaryFunction;

/**
 * SSVD protoptype: non-MR concept verification for Givens QR & SSVD basic algorithms
 * 
 * @author Dmitriy
 *
 */
public class SSVDPrototype {

//	private static final double s_qEpsilon=1e-10;
	
	private Omega 		m_omega;
	private int 		m_kp; // k+p
//	private int 		m_m;
	private GivensThinSolver m_qSolver;
	private double[]	m_yRow;
//	private Vector 		m_yRowV;
	private int 		m_cnt=0, m_blckCnt;
	private int 		m_r;
	private List<UpperTriangular> m_rBlocks = new ArrayList<UpperTriangular>();
	private List<double[][]> m_qtBlocks = new ArrayList<double[][]>();
	private List<double[]>  m_yLookahead;
	

	
	
	public SSVDPrototype(long seed, int kp, int r ) {
		super();
		m_kp=kp;
		m_omega = new Omega(seed, kp/2, kp-(kp/2));
		m_yRow = new double[kp];
//		m_yRowV = new DenseVector(m_yRow,true);
		m_r=r;
		m_yLookahead=new ArrayList<double[]>(m_kp);
	}

	void firstPass ( int aRowId, Vector aRow) 
	throws IOException {

		
		
		m_omega.computeYRow(aRow, m_yRow);
		
		m_yLookahead.add(m_yRow.clone()); // bad for GC but it's just a prototype, hey. in real thing we'll rotate usage of y buff
		
		while ( m_yLookahead.size()>m_kp) { 
			
			if ( m_qSolver == null ) m_qSolver = new GivensThinSolver ( m_r, m_kp);
			
			m_qSolver.appendRow(m_yLookahead.remove(0));
			if ( m_qSolver.isFull()) { 
				UpperTriangular r = m_qSolver.getRTilde();
				double[][] qt = m_qSolver.getThinQtTilde();
				m_qSolver=null; 
				m_qtBlocks.add(qt);
				m_rBlocks.add(r);
			}
			
		}
		m_cnt++;
	}
	
	void finishFirstPass () { 
		
		if ( m_qSolver == null && m_yLookahead.size()==0 ) return; 
		if ( m_qSolver == null ) m_qSolver = new GivensThinSolver (m_yLookahead.size(),m_kp);
		// grow q solver up if necessary
		
		m_qSolver.adjust(m_qSolver.getCnt()+m_yLookahead.size());
		while ( m_yLookahead.size()>0) { 
			
			
			m_qSolver.appendRow(m_yLookahead.remove(0));
			if ( m_qSolver.isFull()) { 
				UpperTriangular r = m_qSolver.getRTilde();
				double[][] qt = m_qSolver.getThinQtTilde();
				m_qSolver=null; 
				m_qtBlocks.add(qt);
				m_rBlocks.add(r);
			}
			
		}
		
		// simulate reducers -- produce qHats 
		for ( int i = 0; i < m_rBlocks.size(); i++ ) 
			m_qtBlocks.set(i,GivensThinSolver.computeQtHat(m_qtBlocks.get(i), i, 
					new _DeepCopyIterator(m_rBlocks.listIterator())
					));
		m_cnt=0;
		m_blckCnt=0;
	}
	
	void secondPass ( int aRowId, Vector aRow,  PartialRowEmitter btEmitter )  
	throws IOException {
		int n=aRow.size();
		double[][] qtHat=m_qtBlocks.get(m_blckCnt);
		
		int r = qtHat[0].length;
		int qRowBlckIndex=r-m_cnt-1; // <-- reverse order since we fed A in reverse
		double[] qRow=new double[m_kp];
		for ( int i =0; i < m_kp; i++ )qRow[i]=qtHat[i][qRowBlckIndex];
		Vector qRowV=new DenseVector(qRow,true);
		
		if ( ++m_cnt==r ) { m_blckCnt++;m_cnt=0; }
		
		for ( int i = 0; i < n; i++ ) 
			btEmitter.emitRow(i, qRowV.times(aRow.getQuick(i)));
		
		
	}
	


	
	
	private static class _DeepCopyIterator implements Iterator<UpperTriangular> {
	
		private Iterator<UpperTriangular> delegate;
		
		public _DeepCopyIterator(Iterator<UpperTriangular> del) {
			super();
			delegate=del;
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

	public static void testThinQr (int dims, int kp, final long rndSeed  ) throws Exception { 
		
		DenseMatrix mx = new DenseMatrix(dims<<2,dims);
		mx.assign(new UnaryFunction() {

			Random m_rnd = new Random(rndSeed);
			
			@Override
			public double apply(double arg0) {
				return m_rnd.nextDouble()*1000;
			}
		});
		
		mx.setQuick(0, 0, 1);
		mx.setQuick(0, 1, 2);
		mx.setQuick(0, 2, 3);
		mx.setQuick(1, 0, 4);
		mx.setQuick(1, 1, 5);
		mx.setQuick(1, 2, 6);
		mx.setQuick(2, 0, 7);
		mx.setQuick(2, 1, 8);
		mx.setQuick(2, 2, 9);
		


		
		SingularValueDecomposition svd2=new SingularValueDecomposition(mx);
		double[] svaluesControl=svd2.getSingularValues();
		
		for ( int i = 0; i < kp; i++ ) 
			System.out.printf("%.3e ", svaluesControl[i]);
		System.out.println();
		
		int m=mx.rowSize(); /*,n=mx.columnSize(); */

		long seed = new Random ( ).nextLong();
		
		final HashMap<Integer, Vector> 
			btRows=new HashMap<Integer,Vector>();
		
		PartialRowEmitter btEmitter = new PartialRowEmitter() {
			@Override
			public void emitRow(int rowNum, Vector row) throws IOException {
				Vector btRow=btRows.get(rowNum);
				if ( btRow != null ) row.addTo(btRow);
				btRows.put(rowNum, btRow==null?new DenseVector(row):btRow);
			}
		};


		SSVDPrototype mapperSimulation=new SSVDPrototype(seed,kp,3000);
		for ( int i =0; i < m; i++ ) 
			mapperSimulation.firstPass(
					i, mx.getRow(i) );
		
		mapperSimulation.finishFirstPass();


		for ( int i  = 0; i < m; i++ ) 
			mapperSimulation.secondPass(i, mx.getRow(i),  btEmitter);
		
//		LocalSSVDTest.assertOrthonormality(mapperSimulation.m_qt.transpose(), false,1e-10);
		
		
		// reconstruct bbt
		final HashMap<Integer, Vector> bbt=new HashMap<Integer, Vector>();
		PartialRowEmitter bbtEmitter = new PartialRowEmitter() {
			
			@Override
			public void emitRow(int rowNum, Vector row) throws IOException {
				Vector bbtRow=bbt.get(rowNum);
				if ( bbtRow != null ) row.addTo(bbtRow);
				bbt.put(rowNum, bbtRow==null?new DenseVector(row):bbtRow);
			}
		};
		
		for ( Map.Entry<Integer, Vector> btRowEntry:btRows.entrySet()) { 
			Vector btRow=btRowEntry.getValue();
			assert btRow.size()==kp;
			for ( int i = 0; i < kp; i++ ) 
				bbtEmitter.emitRow(i, btRow.times(btRow.getQuick(i)));
		}
		
		double[][] bbtValues = new double[kp][];
		for ( int i =0; i < kp; i++ ) { 
			bbtValues[i]=new double[kp];
			Vector bbtRow=bbt.get(i);
			for ( int j=0; j< kp; j++) bbtValues[i][j]=bbtRow.getQuick(j);
		}
		
		EigenDecompositionImpl evd2=new EigenDecompositionImpl(new Array2DRowRealMatrix(bbtValues),0);
		double[] eigenva2=evd2.getRealEigenvalues();
		double[] svalues=new double[kp];
		for ( int i = 0; i < kp; i++ ) 
			svalues[i]=Math.sqrt(eigenva2[i]); // sqrt?
		

		for ( int i = 0; i < kp; i++ ) 
			System.out.printf("%.3e ", svalues[i]);
		System.out.println();
			

	}
	
	public static void testBlockQrWithSSVD ( int dims, int kp, int r, final long rndSeed ) throws Exception { 
		
		DenseMatrix mx = new DenseMatrix(dims<<2,dims);
		mx.assign(new UnaryFunction() {

			Random m_rnd = new Random(rndSeed);
			
			@Override
			public double apply(double arg0) {
				return (m_rnd.nextDouble()-0.5)*1000;
			}
		});
		
		mx.setQuick(0, 0, 1);
		mx.setQuick(0, 1, 2);
		mx.setQuick(0, 2, 3);
		mx.setQuick(1, 0, 4);
		mx.setQuick(1, 1, 5);
		mx.setQuick(1, 2, 6);
		mx.setQuick(2, 0, 7);
		mx.setQuick(2, 1, 8);
		mx.setQuick(2, 2, 9);
		


		
		SingularValueDecomposition svd2=new SingularValueDecomposition(mx);
		double[] svaluesControl=svd2.getSingularValues();
		
		for ( int i = 0; i < kp; i++ ) 
			System.out.printf("%e ", svaluesControl[i]);
		System.out.println();
		
		int m=mx.rowSize();  /*,n=mx.columnSize();*/

		
		final HashMap<Integer, Vector> 
			btRows=new HashMap<Integer,Vector>();
		
		PartialRowEmitter btEmitter = new PartialRowEmitter() {
			@Override
			public void emitRow(int rowNum, Vector row) throws IOException {
				Vector btRow=btRows.get(rowNum);
				if ( btRow != null ) row.addTo(btRow);
				btRows.put(rowNum, btRow==null?new DenseVector(row):btRow);
			}
		};


		SSVDPrototype mapperSimulation=new SSVDPrototype(rndSeed,kp,r);
		for ( int i =0; i < m; i++ ) 
			mapperSimulation.firstPass(
					i, mx.getRow(i) );

		mapperSimulation.finishFirstPass();

		for ( int i  = 0; i < m; i++ ) 
			mapperSimulation.secondPass(i, mx.getRow(i), btEmitter);
		
//		LocalSSVDTest.assertOrthonormality(mapperSimulation.m_qt.transpose(), false,1e-10);
		
		
		// reconstruct bbt
		final HashMap<Integer, Vector> bbt=new HashMap<Integer, Vector>();
		PartialRowEmitter bbtEmitter = new PartialRowEmitter() {
			
			@Override
			public void emitRow(int rowNum, Vector row) throws IOException {
				Vector bbtRow=bbt.get(rowNum);
				if ( bbtRow != null ) row.addTo(bbtRow);
				bbt.put(rowNum, bbtRow==null?new DenseVector(row):bbtRow);
			}
		};
		
		for ( Map.Entry<Integer, Vector> btRowEntry:btRows.entrySet()) { 
			Vector btRow=btRowEntry.getValue();
			assert btRow.size()==kp;
			for ( int i = 0; i < kp; i++ ) 
				bbtEmitter.emitRow(i, btRow.times(btRow.getQuick(i)));
		}
		
		double[][] bbtValues = new double[kp][];
		for ( int i =0; i < kp; i++ ) { 
			bbtValues[i]=new double[kp];
			Vector bbtRow=bbt.get(i);
			for ( int j=0; j< kp; j++) bbtValues[i][j]=bbtRow.getQuick(j);
		}
		
		EigenDecompositionImpl evd2=new EigenDecompositionImpl(new Array2DRowRealMatrix(bbtValues),0);
		double[] eigenva2=evd2.getRealEigenvalues();
		double[] svalues=new double[kp];
		for ( int i = 0; i < kp; i++ ) 
			svalues[i]=Math.sqrt(eigenva2[i]); // sqrt?
		

		for ( int i = 0; i < kp; i++ ) 
			System.out.printf("%e ", svalues[i]);
		System.out.println();
			

	}
	
	public static void main ( String[] args ) throws Exception { 
//		testThinQr();
		long seed=new Random().nextLong();
		testBlockQrWithSSVD(200,200,800, seed);
		testBlockQrWithSSVD(200,20,800, seed);
		testBlockQrWithSSVD(200,20,850, seed); // test trimming
		testBlockQrWithSSVD(200,20,90,seed);
		testBlockQrWithSSVD(200,20,99,seed);
	}
	
}
