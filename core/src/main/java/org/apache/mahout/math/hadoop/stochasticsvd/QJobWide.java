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
import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedList;

import org.apache.hadoop.conf.Configurable;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile.CompressionType;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.compress.DefaultCodec;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.VectorPreprocessor;
import org.apache.mahout.math.VectorWritable;
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
	
	public static class QWidePartitioner extends Partitioner<QWideKeyWritable, VectorWritable> implements Configurable {

	    int m_numMappers;
	    
        @Override
        public int getPartition(QWideKeyWritable key, VectorWritable value,
                int numPartitions) {
            // say we have 10 partitions and 100 mappers.
            // then 0-9 will go to partition 0
            // .. 
            // 90-99 will go to partiton 9.
            return key.m_taskId*numPartitions/m_numMappers;
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
		
		MultipleOutputs.addNamedOutput(job, OUTPUT_QHAT,
		        SequenceFileOutputFormat.class,
		        QWideKeyWritable.class,DenseBlockWritable.class);
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
