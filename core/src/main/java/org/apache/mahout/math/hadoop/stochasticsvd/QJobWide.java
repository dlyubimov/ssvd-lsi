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
import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedList;

import org.apache.hadoop.conf.Configurable;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.SequenceFile.CompressionType;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.compress.DefaultCodec;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.VectorPreprocessor;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.stochasticsvd.QJob.QJobKeyWritable;
import org.apache.mahout.math.hadoop.stochasticsvd.QJob.YPreprocessor;
import org.apache.mahout.math.hadoop.stochasticsvd.QJobWide.QWideKeyWritable;
import org.apache.mahout.math.hadoop.stochasticsvd.io.IOUtil;

/**
 * Compute first level of QHat-transpose blocks. <P>
 * 
 *  This is a "Wide" variation of QJob meaning QHat computations 
 *  are pushed down to reducer to cope with row deficiency issues 
 *  in mappers if rows are too wide to form at least one k+p minimum high 
 *  Q Block in the mapper and increasing minSplitSize parameter is 
 *  not a good option.<P>
 * 
 * See Mahout-376 woking notes for details.
 * 
 * @author dmitriy
 *
 */
public class QJobWide {
	
	public static final String 		 	PROP_OMEGA_SEED="ssvd.omegaseed";
	public static final String 			PROP_K = "ssvd.k";
	public static final String 	        PROP_P = "ssvd.p";
	public static final String          PROP_AROWBLOCK_SIZE="ssvd.arowblock.size";
	
	public static final String 			OUTPUT_R="R";
	public static final String 			OUTPUT_QHAT="QHat";
//	public static final String          OUTPUT_Q="Q";
	public static final String          OUTPUT_Bt="Bt";
	
	
	public static class QWideKeyWritable implements WritableComparable<QWideKeyWritable> {
		
	    private int m_taskId;
	    private int m_taskRowOrdinal;
		

		@Override
		public void readFields(DataInput in) throws IOException {
		    m_taskId=in.readInt();
		    m_taskRowOrdinal = in.readInt();
		}

		@Override
		public void write(DataOutput out) throws IOException {
		    out.writeInt(m_taskId);
		    out.writeInt(m_taskRowOrdinal);
		}
		

		@Override
		public int compareTo(QWideKeyWritable o) {
		    if ( m_taskId < o.m_taskId ) return -1; 
		    else if ( m_taskId > o.m_taskId ) return 1; 
		    if ( m_taskRowOrdinal< o.m_taskRowOrdinal) return -1; 
		    else if ( m_taskRowOrdinal>o.m_taskRowOrdinal) return 1; 
		    return 0; 
		}
		
	}


	
	public static class YWidePreprocessor extends Configured implements VectorPreprocessor {

        private Omega               m_omega;
        private double[]            m_yRow; 
        

        @Override
        public void setConf(Configuration conf) {
            super.setConf(conf);
            if ( conf == null ) return; 
            
            int k = Integer.parseInt(conf.get(PROP_K));
            int p = Integer.parseInt(conf.get(PROP_P));
            m_yRow = new double[k+p];
            long omegaSeed = Long.parseLong(conf.get(PROP_OMEGA_SEED));
            m_omega = new Omega(omegaSeed, k, p);
        }

        @Override
        public boolean beginVector(boolean sequential) {
            Arrays.fill(m_yRow, 0);
            return true;
        }

        @Override
        public void onElement(int index, double value) {
            m_omega.accumDots(index, value, m_yRow);
        }

        @Override
        public void onVectorName(String name) {
            // TODO Auto-generated method stub
        }

        @Override
        public void endVector() {
            // TODO Auto-generated method stub
        } 
	    
	}
	
	public static class QWideMapper extends Mapper<IntWritable, VectorWritable, QWideKeyWritable, VectorWritable> {

	    	    
		private int 				m_kp;
//		private int 				m_r;
		private QWideKeyWritable		m_key = new QWideKeyWritable();
		private VectorWritable      m_value;
		private double[]            m_yRow;
		private LinkedList<Closeable> m_closeables = new LinkedList<Closeable>();
		

		@Override
		protected void map(IntWritable key, VectorWritable value,
				Context context) throws IOException, InterruptedException {

			YWidePreprocessor yp = (YWidePreprocessor)value.getPreprocessor();
			if ( yp.m_yRow!= m_yRow ) { 
			    System.arraycopy(yp.m_yRow, 0, m_yRow, 0, m_kp);
			    yp.m_yRow=m_yRow;
			}
			context.write(m_key, m_value);
			m_key.m_taskRowOrdinal++;
			
		}

		@Override
		protected void setup(final Context context) throws IOException,
				InterruptedException {

			int k = Integer.parseInt(context.getConfiguration().get(PROP_K));
			int p = Integer.parseInt(context.getConfiguration().get(PROP_P));
//			m_r = Integer.parseInt(context.getConfiguration().get(PROP_AROWBLOCK_SIZE));
			m_kp=k+p;
			m_value = new VectorWritable(new DenseVector(m_yRow=new double[m_kp],true));
			context.getConfiguration().set(VectorWritable.PROP_PREPROCESSOR, YWidePreprocessor.class.getName());
			m_key.m_taskId=context.getTaskAttemptID().getTaskID().getId();
			m_key.m_taskRowOrdinal=0;
			
		}

		@Override
		protected void cleanup(Context context) throws IOException,
				InterruptedException {
		    try { 
    			
    			
		    } finally { 			
		        IOUtil.closeAll(m_closeables);
		    }

		} 
	}
	
	public static class QWideTempWritable implements Writable, Iterable<QWideKeyWritable> { 
	    private DenseBlockWritable m_block = new DenseBlockWritable();
	    private int                m_firstRowNum; 
	    private int[]              m_taskIds = new int[24];
	    private int[]              m_taskIdSkips = new int[24];
	    private int                m_idCnt;
	    

	    public DenseBlockWritable getBlock() { return m_block; } 
	    
	    public void reset () { 
	        m_idCnt=0;
	    }
	    
	    public void appendId ( QWideKeyWritable id ) { 
	        int lastTaskId=m_idCnt==0?-1:m_taskIds[m_idCnt-1];
	        if ( lastTaskId==id.m_taskId) { 
	            m_taskIdSkips[m_idCnt-1]++;
	            
	            assert id.m_taskRowOrdinal+1==m_taskIdSkips[m_idCnt-1]; // ordinals must be numbered sequentially
	            
	            return;
	        }
	        
	        if ( m_idCnt+1==m_taskIds .length ) { 
	            m_taskIds = Arrays.copyOf(m_taskIds, m_taskIds.length<<1);
	            m_taskIdSkips=Arrays.copyOf(m_taskIdSkips, m_taskIdSkips.length<<1);
	        }
	        if ( m_idCnt==0) m_firstRowNum=id.m_taskRowOrdinal;
	        m_taskIds[m_idCnt]=id.m_taskId;
	        m_taskIdSkips[m_idCnt]=1;
	        m_idCnt++;
	    }
	    
	    @Override
        public Iterator<QWideKeyWritable> iterator() {
	        final QWideKeyWritable result = new QWideKeyWritable();
	        return new Iterator<QJobWide.QWideKeyWritable>() {

                @Override
                public boolean hasNext() {
                    // TODO Auto-generated method stub
                    return false;
                }

                @Override
                public QWideKeyWritable next() {
                    // TODO Auto-generated method stub
                    return null;
                }

                @Override
                public void remove() {
                    throw new UnsupportedOperationException();
                }
            };
        }

        @Override
        public void readFields(DataInput arg0) throws IOException {
            // TODO Auto-generated method stub
            
        }
        @Override
        public void write(DataOutput arg0) throws IOException {
            // TODO Auto-generated method stub
        }
	    
	    
	    
	    
	}
	
	public static class QWideReducer extends Reducer<QWideKeyWritable, VectorWritable, QWideKeyWritable, VectorWritable> { 
	       
	    private int                 m_kp;
        private ArrayList<double[]> m_yLookahead;
        private GivensThinSolver    m_qSolver;
        private int                 m_blockCnt;
        // private int m_reducerCount;
        private int                 m_r;
        private DenseBlockWritable  m_value = new DenseBlockWritable();
        private QWideKeyWritable     m_key = new QWideKeyWritable();
        private IntWritable         m_tempKey = new IntWritable();
        private MultipleOutputs<QJobKeyWritable, Writable> m_outputs;
        private LinkedList<Closeable> m_closeables = new LinkedList<Closeable>();
        private SequenceFile.Writer m_tempQw;
        private Path m_tempQPath;
        private ArrayList<UpperTriangular> m_rSubseq = new ArrayList<UpperTriangular>();

        private void flushSolver(Context context) throws IOException,
                InterruptedException {
            UpperTriangular r = m_qSolver.getRTilde();
            double[][] qt = m_qSolver.getThinQtTilde();
            m_qSolver = null;

            m_rSubseq.add(new UpperTriangular(r));

            m_value.setBlock(qt);
            m_tempQw.append(m_tempKey, m_value); // this probably should be a
                                                 // sparse row matrix,
            // but compressor should get it for disk and in memory we want it
            // dense anyway, sparse random implementations would be
            // a mostly a memory management disaster consisting of rehashes and
            // GC thrashing. (IMHO)
            m_value.setBlock(null);
        }

        // second pass to run a modified version of computeQHatSequence.
        private void flushQBlocks(Context ctx) throws IOException,
                InterruptedException {
            FileSystem localFs = FileSystem.getLocal(ctx.getConfiguration());
            SequenceFile.Reader m_tempQr = new SequenceFile.Reader(localFs,
                    m_tempQPath, ctx.getConfiguration());
            m_closeables.addFirst(m_tempQr);
            int qCnt = 0;
            while (m_tempQr.next(m_tempKey, m_value)) {
                m_value.setBlock(GivensThinSolver.computeQtHat(
                        m_value.getBlock(),
                        qCnt,
                        new GivensThinSolver.DeepCopyUTIterator(m_rSubseq
                                .iterator())));
                if (qCnt == 1) // just merge r[0] <- r[1] so it doesn't have to
                               // repeat in subsequent computeQHat iterators
                    GivensThinSolver.mergeR(m_rSubseq.get(0),
                            m_rSubseq.remove(1));

                else
                    qCnt++;
                m_outputs.write(OUTPUT_QHAT, m_key, m_value);
            }

            assert m_rSubseq.size() == 1;

            // m_value.setR(m_rSubseq.get(0));
            m_outputs.write(OUTPUT_R, m_key, new VectorWritable(
                    new DenseVector(m_rSubseq.get(0).getData(), true)));

        }
        @Override
        @SuppressWarnings({"rawtypes","unchecked"})
        protected void setup(final Context context) throws IOException,
                InterruptedException {

            int k = Integer.parseInt(context.getConfiguration().get(PROP_K));
            int p = Integer.parseInt(context.getConfiguration().get(PROP_P));
            m_r = Integer.parseInt(context.getConfiguration().get(PROP_AROWBLOCK_SIZE));
            m_kp=k+p;
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
            
            context.getConfiguration().set(VectorWritable.PROP_PREPROCESSOR, YPreprocessor.class.getName());
            
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
	
	public static class QWidePartitioner extends Partitioner<QWideKeyWritable, VectorWritable> implements Configurable {

	    int m_numMappers;
	    
        @Override
        public int getPartition(QWideKeyWritable key, VectorWritable value,
                int numPartitions) {
            return QJobWide.getPartition ( m_numMappers, numPartitions, key.m_taskId);
        }

        @Override
        public void setConf(Configuration conf) {
            if ( conf == null ) return ;
            m_numMappers=conf.getInt("mapred.map.tasks",-1);
            if ( m_numMappers < 0 ) 
                throw new RuntimeException ( "Unable to configure partitioner -- unknown number of map tasks");
            
        }

        @Override
        public Configuration getConf() {
            return null;
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
		job.setJarByClass(QJobWide.class);
		
		
		job.setInputFormatClass(SequenceFileInputFormat.class);
		FileInputFormat.setInputPaths(job, inputPaths);
		if ( minSplitSize>0) 
		    SequenceFileInputFormat.setMinInputSplitSize(job, minSplitSize);
		
		FileOutputFormat.setOutputPath(job, outputPath);
		
//		MultipleOutputs.addNamedOutput(job, OUTPUT_QHAT,
//		        SequenceFileOutputFormat.class,
//		        QWideKeyWritable.class,DenseBlockWritable.class);
		job.getConfiguration().set("mapreduce.output.basename", OUTPUT_QHAT);
		MultipleOutputs.addNamedOutput(job, OUTPUT_R,
		        SequenceFileOutputFormat.class,
		        QWideKeyWritable.class, VectorWritable.class);
		
		SequenceFileOutputFormat.setCompressOutput(job, true);
		SequenceFileOutputFormat.setOutputCompressorClass(job, DefaultCodec.class);
		SequenceFileOutputFormat.setOutputCompressionType(job, CompressionType.BLOCK);
		
		job.setMapOutputKeyClass(QWideKeyWritable.class);
		job.setMapOutputValueClass(VectorWritable.class);
		
		job.setOutputKeyClass(QWideKeyWritable.class);
		job.setOutputValueClass(VectorWritable.class);
		
		job.setMapperClass(QWideMapper.class);
		job.setPartitionerClass(QWidePartitioner.class);
		job.setReducerClass(QWideReducer.class);
		
		job.getConfiguration().setInt(PROP_AROWBLOCK_SIZE,aBlockRows );
		job.getConfiguration().setLong(PROP_OMEGA_SEED, seed);
		job.getConfiguration().setInt(PROP_K, k);
		job.getConfiguration().setInt(PROP_P, p);
		
		// number of reduce tasks doesn't matter. we don't actually 
		// send anything to reducers. in fact, the only reason 
		// we need to configure reduce step is so that combiners can fire.
		// so reduce here is purely symbolic.
		job.setNumReduceTasks(numReduceTasks);
		
		job.submit();
		job.waitForCompletion(false);
		
		if ( !job.isSuccessful())
			throw new IOException ( "Q job unsuccessful.");
		
		
		
	}

	public static enum QJobCntEnum { 
		NUM_Q_BLOCKS;
	}

    public static int getPartition ( int numMapperTasks, int numPartitions, int forTaskId ) { 
        // say we have 10 partitions and 100 mappers.
        // then 0-9 will go to partition 0
        // .. 
        // 90-99 will go to partiton 9.
        return forTaskId*numPartitions/numMapperTasks;
    }
	
}
