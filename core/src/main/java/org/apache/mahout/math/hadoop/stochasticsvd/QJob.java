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
import java.io.DataInput;
import java.io.DataOutput;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.CompressionType;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.compress.DefaultCodec;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.stochasticsvd.io.IOUtil;

/**
 * Compute first level of QHat-transpose blocks. 
 * 
 * See Mahout-376 woking notes for details.
 * 
 * @author dmitriy
 *
 */
public class QJob {
	
	public static final String 		 	PROP_OMEGA_SEED="ssvd.omegaseed";
	public static final String 			PROP_K = "ssvd.k";
	public static final String 	        PROP_P = "ssvd.p";
	public static final String          PROP_AROWBLOCK_SIZE="ssvd.arowblock.size";
	
	public static final String 			OUTPUT_R="R";
	public static final String 			OUTPUT_QHAT="QHat";
//	public static final String          OUTPUT_Q="Q";
	public static final String          OUTPUT_Bt="Bt";
	
	public static class QJobKeyWritable implements WritableComparable<QJobKeyWritable> {
		
	    private int taskId;
	    private int taskRowOrdinal;
		

		@Override
		public void readFields(DataInput in) throws IOException {
		    taskId=in.readInt();
		    taskRowOrdinal = in.readInt();
		}

		@Override
		public void write(DataOutput out) throws IOException {
		    out.writeInt(taskId);
		    out.writeInt(taskRowOrdinal);
		}
		

		@Override
		public int compareTo(QJobKeyWritable o) {
		    if ( taskId < o.taskId ) return -1; 
		    else if ( taskId > o.taskId ) return 1; 
		    if ( taskRowOrdinal< o.taskRowOrdinal) return -1; 
		    else if ( taskRowOrdinal>o.taskRowOrdinal) return 1; 
		    return 0; 
		}
		



		
	}


	public static class QMapper extends Mapper<IntWritable, VectorWritable, QJobKeyWritable, VectorWritable> {
		
		private int 				m_kp;
		private Omega 				m_omega;
		private ArrayList<double[]> m_yLookahead;
		private GivensThinSolver 	m_qSolver;
		private int 				m_blockCnt;
//		private int 				m_reducerCount;
		private int 				m_r;
		private DenseBlockWritable	m_value = new DenseBlockWritable();
		private QJobKeyWritable		m_key = new QJobKeyWritable();
		private IntWritable         m_tempKey = new IntWritable();
		private MultipleOutputs<QJobKeyWritable, Writable>     m_outputs;
		private LinkedList<Closeable> m_closeables = new LinkedList<Closeable>();
		private SequenceFile.Writer   m_tempQw;
		private Path                  m_tempQPath;
		private ArrayList<UpperTriangular> m_rSubseq = new ArrayList<UpperTriangular>();

		private void flushSolver ( Context context ) throws IOException, InterruptedException { 
			 UpperTriangular r = m_qSolver.getRTilde();
			 double[][]qt = m_qSolver.getThinQtTilde();
			 m_qSolver =null;

			 m_rSubseq.add(new UpperTriangular(r));
			 
			 m_value.setBlock(qt);
			 m_tempQw.append(m_tempKey, m_value); // this probably should be a sparse row matrix, 
			                         // but compressor should get it for disk and in memory we want it 
			                         // dense anyway, sparse random implementations would be 
			                         // a mostly a  memory management disaster consisting of rehashes and GC thrashing. (IMHO)
			 m_value.setBlock(null);
		}
		
		// second pass to run a modified version of computeQHatSequence.
		private void flushQBlocks (Context ctx ) throws IOException, InterruptedException { 
		    FileSystem localFs = FileSystem.getLocal(ctx.getConfiguration());
		    SequenceFile.Reader m_tempQr = new SequenceFile.Reader(localFs, m_tempQPath, ctx.getConfiguration());
		    m_closeables.addFirst(m_tempQr);
		    int qCnt = 0; 
		    while ( m_tempQr.next(m_tempKey,m_value)) { 
                m_value.setBlock(GivensThinSolver.computeQtHat(
                        m_value.getBlock(), qCnt, 
                        new GivensThinSolver.DeepCopyUTIterator(m_rSubseq.iterator())));
                if ( qCnt == 1 ) // just merge r[0] <- r[1] so it doesn't have to repeat in subsequent computeQHat iterators
                    GivensThinSolver.mergeR(m_rSubseq.get(0), m_rSubseq.remove(1));
                    
                else qCnt++;
                m_outputs.write(OUTPUT_QHAT, m_key, m_value);
		    }
		    
		    assert m_rSubseq.size()==1;

//		    m_value.setR(m_rSubseq.get(0));
	        m_outputs.write(OUTPUT_R, m_key, new VectorWritable(new DenseVector(m_rSubseq.get(0).getData(),true)));
	        

		}

		@Override
		protected void map(IntWritable key, VectorWritable value,
				Context context) throws IOException, InterruptedException {
			double[] yRow=null;
			if ( m_yLookahead.size()==m_kp) { 
				 yRow= m_yLookahead.remove(0);
				 if ( m_qSolver == null ) m_qSolver = new GivensThinSolver(m_r, m_kp);
				 m_qSolver.appendRow(yRow);
				 if ( m_qSolver.isFull()) { 
					 
					 flushSolver(context);
					 m_blockCnt++;
				 }
			} else yRow = new double[m_kp];
			m_omega.computeYRow(value.get(), yRow);
			m_yLookahead.add(yRow);
		}

		@Override
		@SuppressWarnings({"rawtypes","unchecked"})
		protected void setup(final Context context) throws IOException,
				InterruptedException {

			int k = Integer.parseInt(context.getConfiguration().get(PROP_K));
			int p = Integer.parseInt(context.getConfiguration().get(PROP_P));
			m_kp=k+p;
			long omegaSeed = Long.parseLong(context.getConfiguration().get(PROP_OMEGA_SEED));
			m_r = Integer.parseInt(context.getConfiguration().get(PROP_AROWBLOCK_SIZE));
			m_omega = new Omega(omegaSeed, k, p);
			m_yLookahead=new ArrayList<double[]>(m_kp);
			m_outputs=new MultipleOutputs(context);
			m_closeables.addFirst(new Closeable() {
                @Override
                public void close() throws IOException {
                    try { 
                        m_outputs.close();
                    } catch ( InterruptedException exc ) { 
                        throw new IOException ( exc );
                    }
                }
            });
			
			// temporary Q output 
			// hopefully will not exceed size of IO cache in which case it is only good since it 
			// is going to be maanged by kernel, not java GC. And if IO cache is not good enough, 
			// then at least it is always sequential.
			String taskTmpDir = System.getProperty("java.io.tmpdir");
			FileSystem localFs=FileSystem.getLocal(context.getConfiguration());
			m_tempQPath = new Path ( new Path ( taskTmpDir), "q-temp.seq");
			m_tempQw=SequenceFile.createWriter(localFs, 
			        context.getConfiguration(), 
			        m_tempQPath, 
			        IntWritable.class, 
			        DenseBlockWritable.class,
			        CompressionType.BLOCK );
			m_closeables.addFirst(m_tempQw);
			m_closeables.addFirst(new IOUtil.DeleteFileOnClose(new File( m_tempQw.toString())));
			
			
		}

		@Override
		protected void cleanup(Context context) throws IOException,
				InterruptedException {
		    try { 
    			if ( m_qSolver == null && m_yLookahead.size()==0 ) return; 
    			if ( m_qSolver == null ) m_qSolver = new GivensThinSolver (m_yLookahead.size(),m_kp);
    			// grow q solver up if necessary
    			
    			m_qSolver.adjust(m_qSolver.getCnt()+m_yLookahead.size());
    			while ( m_yLookahead.size()>0) { 
    				
    				m_qSolver.appendRow(m_yLookahead.remove(0));
    				if ( m_qSolver.isFull()) { 
    					flushSolver(context);
    					m_blockCnt++;
    				}
    				
    			}
    			m_closeables.remove(m_tempQw);
    			m_tempQw.close();
    			flushQBlocks(context);
    			
    			
		    } finally { 			
		        IOUtil.closeAll(m_closeables);
		    }

		} 
	}
	
	
	
	public static void run ( Configuration conf, 
			Path[] inputPaths, 
			Path outputPath,
			int aBlockRows,
			int minSplitSize,
			int k,
			int p,
			long seed ,
			int numReduceTasks  ) 
	throws ClassNotFoundException, InterruptedException, IOException {
		
		Job job=new Job(conf);
		job.setJobName("Q-job");
		job.setJarByClass(QJob.class);
		
		
		job.setInputFormatClass(SequenceFileInputFormat.class);
		FileInputFormat.setInputPaths(job, inputPaths);
		if ( minSplitSize>0) 
		    SequenceFileInputFormat.setMinInputSplitSize(job, minSplitSize);
		
		FileOutputFormat.setOutputPath(job, outputPath);
		
		MultipleOutputs.addNamedOutput(job, OUTPUT_QHAT,
		        SequenceFileOutputFormat.class,
		        QJobKeyWritable.class,DenseBlockWritable.class);
		MultipleOutputs.addNamedOutput(job, OUTPUT_R,
		        SequenceFileOutputFormat.class,
		        QJobKeyWritable.class, VectorWritable.class);
		
		SequenceFileOutputFormat.setCompressOutput(job, true);
		SequenceFileOutputFormat.setOutputCompressorClass(job, DefaultCodec.class);
		SequenceFileOutputFormat.setOutputCompressionType(job, CompressionType.BLOCK);
		
		job.setMapOutputKeyClass(QJobKeyWritable.class);
		job.setMapOutputValueClass(VectorWritable.class);
		
		job.setOutputKeyClass(QJobKeyWritable.class);
		job.setOutputValueClass(VectorWritable.class);
		
		job.setMapperClass(QMapper.class);
		
		job.getConfiguration().setInt(PROP_AROWBLOCK_SIZE,aBlockRows );
		job.getConfiguration().setLong(PROP_OMEGA_SEED, seed);
		job.getConfiguration().setInt(PROP_K, k);
		job.getConfiguration().setInt(PROP_P, p);
		
		// number of reduce tasks doesn't matter. we don't actually 
		// send anything to reducers. in fact, the only reason 
		// we need to configure reduce step is so that combiners can fire.
		// so reduce here is purely symbolic.
		job.setNumReduceTasks(0 /*numReduceTasks*/);
		
		job.submit();
		job.waitForCompletion(false);
		
		if ( !job.isSuccessful())
			throw new IOException ( "Q job unsuccessful.");
		
		
		
	}

	public static enum QJobCntEnum { 
		NUM_Q_BLOCKS;
	}
	
}
