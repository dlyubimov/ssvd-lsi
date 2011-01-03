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
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.EigenDecompositionImpl;
import org.apache.commons.math.linear.RealMatrix;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.SequenceFile.CompressionType;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.stochasticsvd.io.IOUtil;

/**
 * Stochastic SVD solver. <P>
 * 
 * Implementation details are in my working notes in 
 * MAHOUT-376 (https://issues.apache.org/jira/browse/MAHOUT-376). <P>
 * 
 * This class is central public API for SSVD solver. 
 * The use pattern is as follows: 
 * 
 * <UL>
 * <LI> create the solver using constructor and supplying computation parameters. 
 * <LI> set optional parameters thru setter methods. 
 * <LI> call {@link #run()}. 
 * <LI> {@link #getUPath()} (if computed) returns the path to the directory containing m x k U matrix file(s).
 * <LI> {@link #getVPath()} (if computed) returns the path to the directory containing n x k V matrix file(s).
 * 
 * </UL>
 * 
 * 
 * @author Dmitriy
 *
 */
public class SSVDSolver {
	
	

	private double[] m_svalues;
	private boolean m_computeU=true, m_computeV =true;
	private String m_uPath;
	private String m_vPath;
	
	// configured stuff 
	private    Configuration   m_conf; 
	private    Path[]          m_inputPath; 
	private    Path            m_outputPath; 
	private    int             m_ablockRows;
	private    int             m_k, m_p;
	private    int             m_reduceTasks; 
	private    int             m_minSplitSize = -1;
	private    boolean         m_wide;
	private    boolean         m_cUHalfSigma=false;
	private    boolean         m_cVHalfSigma=false;
	
	
	/**
	 * create new SSVD solver. Required parameters are passed to 
	 * constructor to ensure they are set. Optional parameters can 
	 * be set using setters . <P>
	 * 
	 * @param conf hadoop configuration
	 * @param inputPath Input path (should be compatible with DistributedRowMatrix as of the time of this writing).
	 * @param outputPath Output path containing U, V and singular values vector files.
	 * @param ablockRows The vertical hight of a q-block (bigger value require more memory in 
	 *         mappers+ perhaps larger <code>minSplitSize</code> values
	 * @param k desired rank
	 * @param p SSVD oversampling parameter
	 * @param reduceTasks Number of reduce tasks (where applicable)
	 * @throws IOException when IO condition occurs.
	 */
	
	public SSVDSolver(Configuration conf, Path[] inputPath, Path outputPath, 
			int ablockRows, int k, int p, int reduceTasks, boolean wide) throws IOException {
	    m_conf = conf; 
	    m_inputPath = inputPath;
	    m_outputPath = outputPath;
	    m_ablockRows = ablockRows; 
	    m_k = k; 
	    m_p = p; 
	    m_reduceTasks = reduceTasks;
	    m_wide = wide;
	}

	
	
    
    public void setcUHalfSigma(boolean cUHat) {
        this.m_cUHalfSigma = cUHat;
    }

    public void setcVHalfSigma(boolean cVHat) {
        this.m_cVHalfSigma = cVHat;
    }

    /**
	 * The setting controlling whether to compute U matrix of low rank SSVD.
	 * 
	 */
	public void setComputeU(boolean val ) { 
	    m_computeU = val; 
	}
	/**
	 * Setting controlling whether to compute V matrix of low-rank SSVD.
	 * @param val true if we want to output V matrix. Default is true. 
	 */
	public void setComputeV ( boolean val ) { 
	    m_computeV = val; 
	}
	
	/**
	 * Sometimes, if requested A blocks become larger than a split, 
	 * we may need to use that to ensure at least k+p rows of A 
	 * get into a split. This is requirement necessary to 
	 * obtain orthonormalized Q blocks of SSVD.
	 * 
	 * @param size the minimum split size to use
	 */
	public void setMinSplitSize ( int size ) { 
	    m_minSplitSize=size;
	}

	/**
	 * This contains k+p singular values resulted from the solver run. 
	 * 
	 * @return singlular values (largest to smallest) 
	 */
	public double[] getSingularValues () { 
		return m_svalues;
	}
	
	/**
	 * returns U path (if computation were requested and successful).
	 * 
	 * @return U output hdfs path, or null if computation was not completed for whatever reason.
	 */
	public String getUPath () { 
	    return m_uPath;
	}
	
	/**
	 * return V path ( if computation was requested and successful ) . 
	 * 
	 * @return V output hdfs path, or null if computation was not completed for whatever reason.
	 */
	public String getVPath () { 
	    return m_vPath;
	}

	/** 
	 * run all SSVD jobs. 
	 * 
	 * @throws IOException if I/O condition occurs.
	 */
	public void run() throws IOException  {
		
		LinkedList<Closeable> closeables = new LinkedList<Closeable>();
		try {
		    Class<?extends Writable> labelType = _sniffInputLabelType(m_inputPath, m_conf, closeables);
		    FileSystem fs = FileSystem.get(m_conf);
		    
			
			Path qPath = new Path(m_outputPath, "Q-job");
			Path btPath = new Path ( m_outputPath, "Bt-job" );
			Path bbtPath = new Path ( m_outputPath, "BBt-job");
			Path uHatPath = new Path ( m_outputPath, "UHat");
			Path svPath = new Path ( m_outputPath, "Sigma");
			Path uPath = new Path ( m_outputPath, "U");
			Path vPath = new Path ( m_outputPath, "V");
			
			fs.delete(qPath, true); // or we can't re-run it repeatedly, just in case.
			fs.delete(btPath, true);
			fs.delete(bbtPath,true);
			fs.delete(uHatPath, true);
			fs.delete(svPath, true);
			fs.delete(uPath, true ); 
			fs.delete(vPath, true );
			
			Random rnd = new Random();
			long seed = rnd.nextLong();
			if ( ! m_wide ) 
			    QJob.run(m_conf, m_inputPath, qPath,
    			        m_ablockRows,
    			        m_minSplitSize,
    			        m_k,m_p,seed,m_reduceTasks);
			else 
	             QJobWide.run(m_conf, m_inputPath, qPath,
	                     m_ablockRows,
	                     m_minSplitSize,
	                     m_k,m_p,seed,m_reduceTasks);

			BtJob.run(
			        m_conf, 
			        m_inputPath, 
			        qPath, 
			        btPath, 
			        m_minSplitSize,
			        m_k, m_p, 
			        m_reduceTasks,
			        labelType
			        );
			
			BBtJob.run(m_conf,new Path( btPath, 
			        BtJob.OUTPUT_Bt+"-*"), 
			        bbtPath, 1);
			
			double[][] bbt=loadDistributedRowMatrix(fs, 
			        new Path(bbtPath,BBtJob.OUTPUT_BBt+"-*"), m_conf);
			
			// make sure it is symmetric exactly. sometimes the commons math package detects 
			// rounding error and barfs at it. 
			for ( int i = 0; i < m_k+m_p-1; i++ )
				for ( int j = i+1; j < m_k+m_p; j++ ) 
					bbt[j][i]=bbt[i][j];
			
			m_svalues=new double[m_k+m_p];
			
			// try something else.
			EigenDecompositionImpl evd2=new EigenDecompositionImpl(new Array2DRowRealMatrix(bbt),0);
			double[] eigenva2=evd2.getRealEigenvalues();
			for ( int i = 0; i < m_k+m_p; i++ ) 
				m_svalues[i]=Math.sqrt(eigenva2[i]); // sqrt?
			
			// save/redistribute UHat 
			//
			RealMatrix uHatrm = evd2.getV();
			
			fs.mkdirs(uHatPath);
			SequenceFile.Writer uHatWriter = SequenceFile.createWriter(
			        fs, m_conf, 
			        uHatPath=new Path ( uHatPath, "uhat.seq"),
			        IntWritable.class,
			        VectorWritable.class,
			        CompressionType.BLOCK );
			closeables.addFirst(uHatWriter);
			
		    int m=uHatrm.getRowDimension();
		    IntWritable iw = new IntWritable(); 
		    VectorWritable vw = new VectorWritable();
		    for ( int i = 0; i < m; i ++ ) { 
		        vw.set(new DenseVector (uHatrm.getRow(i), true ));
		        iw.set(i);
		        uHatWriter.append(iw,vw);
		    }
		    
		    closeables.remove(uHatWriter);
			uHatWriter.close();
			
            SequenceFile.Writer svWriter = SequenceFile.createWriter(
                    fs, m_conf, 
                    svPath = new Path ( svPath, "svalues.seq"),
                    IntWritable.class,
                    VectorWritable.class,
                    CompressionType.BLOCK );
            
            closeables.addFirst(svWriter);
            
            vw.set(new DenseVector(m_svalues,true));
            svWriter.append(iw, vw);

            closeables.remove(svWriter);
            svWriter.close();
			        
			UJob ujob =null;
			VJob vjob = null;
			if ( m_computeU ) (ujob= new UJob() ).start(
			        m_conf, 
			        new Path(btPath,BtJob.OUTPUT_Q+"-*"), 
			        uHatPath,
			        svPath,
			        uPath, 
			        m_k,
			        m_reduceTasks,
			        labelType,
			        m_cUHalfSigma
			         ); // actually this is map-only job anyway
			
			if ( m_computeV ) (vjob=new VJob()).start ( 
			        m_conf, 
			        new Path ( btPath, BtJob.OUTPUT_Bt+"-*"),
			        uHatPath,
			        svPath,
			        vPath,
			        m_k,
			        m_reduceTasks,
			        m_cVHalfSigma
			        );
			
			if ( ujob != null ) { 
			    ujob.waitForCompletion();
			    m_uPath=uPath.toString();
			}
			if( vjob != null ) { 
			    vjob.waitForCompletion();
			    m_vPath=vPath.toString();
			}
			
			
		} catch ( InterruptedException exc ) { 
			throw new IOException ( "Interrupted",exc );
		} catch ( ClassNotFoundException exc ) { 
			throw new IOException ( exc );
			
		} finally { 
			IOUtil.closeAll(closeables);
		}
		
	}
	
	private  static Class<?extends Writable> _sniffInputLabelType ( 
	        Path[] inputPath, Configuration conf, LinkedList<Closeable> closeables ) throws IOException { 
	    FileSystem fs = FileSystem.get(conf);
	    for ( Path p:inputPath) { 
	        FileStatus fstats[]=fs.globStatus(p);
	        if ( fstats==null||fstats.length==0 ) continue; 
	        SequenceFile.Reader r= new SequenceFile.Reader(fs, fstats[0].getPath(), conf);
	        closeables.addFirst(r);
	        
	        try { 
	            return r.getKeyClass().asSubclass(Writable.class);
	        } finally { 
	            closeables.remove(r);
	            r.close();
	        }
	    }
	    
	    throw new IOException ("Unable to open input files to determine input label type.");
	}
	
	
    private static final Pattern s_outputFilePattern = Pattern.compile("(\\w+)-(m|r)-(\\d+)(\\.\\w+)?");

	public static Comparator<FileStatus> s_partitionComparator = new Comparator<FileStatus>() {
        private Matcher m_matcher = s_outputFilePattern.matcher("");
        
        @Override
        public int compare(FileStatus o1, FileStatus o2) {
            m_matcher.reset(o1.getPath().getName());
            if ( ! m_matcher.matches()) throw new RuntimeException ( String.format ( 
                    "Unexpected file name, unable to deduce partition #:%s",
                    o1.getPath().toString()));
            int p1 = Integer.parseInt(m_matcher.group(3));
            m_matcher.reset(o2.getPath().getName());
            if ( ! m_matcher.matches()) throw new RuntimeException ( String.format ( 
                    "Unexpected file name, unable to deduce partition #:%s",
                    o2.getPath().toString()));
                    
            int p2 = Integer.parseInt(m_matcher.group(3));
            return p1-p2;
        }
        

    };
	
    /**
     * helper capabiltiy to load distributed row matrices into dense matrix 
     * (to support tests mainly). 
     * 
     * @param fs filesystem
     * @param glob FS glob
     * @param conf configuration
     * @return Dense matrix array
     * @throws IOException when I/O occurs. 
     */
	public static double[][] loadDistributedRowMatrix(FileSystem fs, Path glob, Configuration conf ) throws IOException { 
		
		FileStatus[] files=fs.globStatus(glob);
		if ( files == null ) return null;
		
		ArrayList<double[]> denseData = new ArrayList<double[]>();
		IntWritable iw = new IntWritable();
		VectorWritable vw = new VectorWritable();
		
//		int m=0;
	
		// assume it is partitioned output, so we need to read them up 
		// in order of partitions.
		Arrays.sort(files, s_partitionComparator);
		
		for ( FileStatus fstat:files)  {
			Path file = fstat.getPath();
			SequenceFile.Reader reader = new SequenceFile.Reader(fs, file, conf);
			try { 
				while ( reader.next(iw,vw)) {
					Vector v= vw.get();
					int size;
					double[] row=new double[size=v.size()];
					for ( int i = 0; i < size; i++) row[i]=v.get(i);
					// ignore row label. 
					// int rowIndex=iw.get();
					denseData.add(row);
					
				}
			} finally { 
				reader.close();
			}
		}
		
		return denseData.toArray(new double[denseData.size()][]);
	}
	
}
