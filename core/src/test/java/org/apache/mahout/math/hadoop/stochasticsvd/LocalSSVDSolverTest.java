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

import java.io.Closeable;
import java.io.File;
import java.util.LinkedList;
import java.util.Random;

import junit.framework.Assert;
import junit.framework.TestCase;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.CompressionType;
import org.apache.hadoop.io.compress.DefaultCodec;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.SingularValueDecomposition;
import org.apache.mahout.math.VectorWritable;

public class LocalSSVDSolverTest extends TestCase {
	
	
	private static final double s_epsilon = 1.0E-5d;
	
	
	public void testSSVDSolver() throws Exception {
		
		Configuration conf = new Configuration();
		conf.set("mapred.job.tracker","local");
		conf.set("fs.default.name","file:///");
		
//		conf.set("mapred.job.tracker","localhost:11011");
//		conf.set("fs.default.name","hdfs://localhost:11010/");
		
		
		LinkedList<Closeable> closeables=new LinkedList<Closeable>();
		Random rnd = new Random();
		int m = 10000, n=100, k=40, p=60, ablockRows=10000;
		
		double muAmplitude=5e+1;
		
		File tmpDir = new File("svdtmp");
		tmpDir.mkdir();
        conf.set("hadoop.tmp.dir", tmpDir.getAbsolutePath());

		File aDir=new File (tmpDir, "A");
		aDir.mkdir();
		
		Path aLocPath = new Path(new Path ( aDir.getAbsolutePath()), "A.seq");
		
		// create distributed row matrix-like struct
		SequenceFile.Writer w= SequenceFile.createWriter(
				FileSystem.getLocal(conf), 
				conf, 
				aLocPath, 
				IntWritable.class, 
				VectorWritable.class,
				CompressionType.BLOCK,
				new DefaultCodec()
				);
		closeables.addFirst(w);
		
		double[] row= new double[n];
		DenseVector dv = new DenseVector(row, true);
		VectorWritable vw  = new VectorWritable(dv);
		IntWritable roww=new IntWritable();
		
		for( int i = 0; i < m; i++ ) { 
			for ( int j = 0; j < n; j++ ) 
				row[j]=muAmplitude*(rnd.nextDouble()-0.5);
			roww.set(i);
			w.append(roww, vw);
		}
		closeables.remove(w);
		w.close();
		

		FileSystem fs = FileSystem.get(conf);
		
		Path tempDirPath=new Path (fs.getWorkingDirectory(), "svd-proc");
		Path aPath=new Path ( tempDirPath, "A/A.seq");
		fs.copyFromLocalFile(aLocPath, aPath);
		
		Path svdOutPath=new Path(tempDirPath,"SSVD-out");
		
		// make sure we wipe out previous test results, just a convenience 
		fs.delete(svdOutPath,true);
		
		

		SSVDSolver ssvd=new SSVDSolver(conf,
				new Path[] { aPath }, 
				svdOutPath,
				ablockRows,k,p,
				3,
				false);
		ssvd.run();

	    double[] stochasticSValues=ssvd.getSingularValues();
	    System.out.println ("--SSVD solver singular values:");
        dumpSv(stochasticSValues);
        System.out.println ("--Colt SVD solver singular values:");

		// try to run the same thing without stochastic algo
		double[][] a=SSVDSolver.loadDistributedRowMatrix(fs, aPath, conf);
		
//		SingularValueDecompositionImpl svd=new SingularValueDecompositionImpl(new Array2DRowRealMatrix(a));
		SingularValueDecomposition svd2=new SingularValueDecomposition(new DenseMatrix(a));
		
		a=null;
		
		
		double[] svalues2=svd2.getSingularValues();
		dumpSv(svalues2);
		
		for (int i =0; i < k+p;i++ )
		    Assert.assertTrue(Math.abs(svalues2[i]- stochasticSValues[i])<=s_epsilon);

		double[][] q=  SSVDSolver.loadDistributedRowMatrix(fs, 
				new Path ( svdOutPath, "Bt-job/"+BtJob.OUTPUT_Q+"-*"), conf);
		
		SSVDPrototypeTest.assertOrthonormality(new DenseMatrix(q), false,s_epsilon);
		
	}
	
	static void dumpSv (double[] s ) { 
		System.out.printf("svs: ");
		for (int i = 0; i < s.length;i++) 
			System.out.printf("%f  ", s[i]);
		System.out.println();
		
	}
	static void dump (double[][] matrix) { 
		for ( int i = 0; i < matrix.length; i++ )  {
			for ( int j = 0; j < matrix[i].length; j++  ) 
				System.out.printf("%f  ", matrix[i][j]);
			System.out.println();
		}
	}

}
