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
import java.util.Arrays;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.CompressionType;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.compress.DefaultCodec;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.stochasticsvd.QJob.QJobKeyWritable;

public class BtJob {
    
    	
    
    public static final String  OUTPUT_Q = "Q";
    public static final String  OUTPUT_Bt = "Bt";
    public static final String  PROP_QJOB_PATH = "ssvd.QJob.path";

    	
    
    
	public static class BtMapper extends Mapper<Writable, VectorWritable, IntWritable, VectorWritable> {

        private SequenceFile.Reader m_qInput;
        private ArrayList<UpperTriangular> m_Rs = new ArrayList<UpperTriangular>();
        private int         m_blockNum;
        private double[][]  m_qt;
        private int         m_cnt=0, m_r;
        private MultipleOutputs<IntWritable,VectorWritable> m_outputs;
        private IntWritable m_btKey = new IntWritable();
        private VectorWritable m_btValue =new VectorWritable();
        private int         m_kp;
        private VectorWritable m_qRowValue = new VectorWritable();
        private int         m_qCount; // debug
        
        
        void loadNextQt (Context ctx ) throws IOException, InterruptedException { 
            QJobKeyWritable key = new QJobKeyWritable();
            DenseBlockWritable v = new DenseBlockWritable();
            
            
            boolean more=m_qInput.next(key, v);
            assert more;
            
            m_qt=GivensThinSolver.computeQtHat(v.getBlock(), m_blockNum==0?0:1,
                    new GivensThinSolver.DeepCopyUTIterator(m_Rs.iterator()));
            m_r= m_qt[0].length;
            m_kp=m_qt.length;
            if ( m_btValue.get()==null ) m_btValue.set(new DenseVector(m_kp));
            if ( m_qRowValue.get()==null ) m_qRowValue.set( new DenseVector ( m_kp));
            
            // also output QHat -- although we don't know the A labels there. Is it important?
//            DenseVector qRow = new DenseVector(m_kp);
//            IntWritable oKey = new IntWritable();
//            VectorWritable oV = new VectorWritable();
//            
//            for ( int i = m_r-1; i>=0; i-- ) { 
//                for ( int j= 0; j < m_kp; j++ )
//                    qRow.setQuick(j, m_qt[j][i]);
//                oKey.set((m_blockNum<<20)+m_r-i-1); 
//                oV.set(qRow);
//                // so the block #s range is thus 0..2048, and number of rows per block is 0..2^20.
//                // since we are not really sending it out to sort (it is a 'side file'), 
//                // it doesn't matter if it overflows.
//                m_outputs.write( OUTPUT_Q, oKey, oV);
//            }
            m_qCount++;
        }
        
        @Override
        protected void cleanup(
                Context context)
                throws IOException, InterruptedException {

            if ( m_qInput != null ) m_qInput.close();
            if ( m_outputs != null ) m_outputs.close();
            super.cleanup(context);
        }

        @Override
        protected void map(Writable key, VectorWritable value,
                Context context)
                throws IOException, InterruptedException {
            if ( m_qt != null && m_cnt++==m_r ) m_qt=null;
            if ( m_qt==null ) { loadNextQt(context ); m_cnt = 1; }
            
            // output Bt outer products
            Vector aRow = value.get();
            int qRowIndex=m_r-m_cnt; // because QHats are initially stored in reverse
            Vector qRow = m_qRowValue.get();
            for ( int j = 0; j < m_kp; j++ ) 
                qRow.setQuick(j, m_qt[j][qRowIndex]);
            
            m_outputs.write(OUTPUT_Q, key, m_qRowValue); // make sure Qs are inheriting A row labels.
            
            int n=aRow.size();
            Vector m_btRow = m_btValue.get();
            for ( int i =0; i < n; i++ ) { 
                double mul=aRow.getQuick(i);
                for ( int j = 0; j< m_kp; j++ ) 
                    m_btRow.setQuick(j, mul*qRow.getQuick(j));
                m_btKey.set(i);
                context.write(m_btKey, m_btValue);
            }
            
        }

        
        @Override
        protected void setup(Context context)
                throws IOException, InterruptedException {
            super.setup(context);
            
            Path qJobPath=new Path(context.getConfiguration().get(PROP_QJOB_PATH));
            
            FileSystem fs = FileSystem.get(context.getConfiguration());
            // actually this is kind of dangerous
            // becuase this routine thinks we need to create file name for 
            // our current job and this will use -m- so it's just serendipity we are calling 
            // it from the mapper too as the QJob did.
            Path qInputPath= new Path ( qJobPath, FileOutputFormat.getUniqueFile(context, QJob.OUTPUT_QHAT, ""));
            m_qInput=new SequenceFile.Reader(fs, qInputPath, context.getConfiguration());
            
            m_blockNum = context.getTaskAttemptID().getTaskID().getId();
            
            // read all r files _in order of task ids_, i.e. partitions
            Path rPath = new Path ( qJobPath, QJob.OUTPUT_R+"-*");
            FileStatus[] rFiles = fs.globStatus(rPath);
            
            if ( rFiles == null ) 
                throw new IOException ( "Can't find R inputs ");
            
            Arrays.sort(rFiles, SSVDSolver.s_partitionComparator);
            
            QJobKeyWritable rKey = new QJobKeyWritable();
            VectorWritable rValue = new VectorWritable();
            
            int block=0;
            for ( FileStatus fstat:rFiles ) { 
                SequenceFile.Reader rReader=new SequenceFile.Reader(fs,fstat.getPath(),context.getConfiguration());
                try { 
                    rReader.next(rKey, rValue);
                } finally { 
                    rReader.close();
                }
                if ( block<m_blockNum&&block>0)
                    GivensThinSolver.mergeR(m_Rs.get(0), new UpperTriangular(rValue.get()));
                else m_Rs.add(new UpperTriangular ( rValue.get()));
                block++;
            }
            m_outputs = new MultipleOutputs<IntWritable, VectorWritable>(context);
        } 
	}
	


	public static class OuterProductReducer extends Reducer<IntWritable, VectorWritable, IntWritable, VectorWritable> {
	    
	    private VectorWritable m_oValue = new VectorWritable();

        @Override
        protected void reduce(IntWritable key, Iterable<VectorWritable> values,
                Context ctx)
                throws IOException, InterruptedException {
            Iterator<VectorWritable> vwIter= values.iterator();
            DenseVector accum=new DenseVector(vwIter.next().get()); // so it will reallocate 
                                                              // for each iteration... ideally we may want
                                                              // to optimize that a little to prevent GC thrashing
            while ( vwIter.hasNext()) accum.addAll(vwIter.next().get());
            m_oValue.set(accum);
            ctx.write(key,m_oValue);
        } 
	    
	}
	
	
    public static void run ( Configuration conf, 
            Path inputPathA[],
            Path inputPathQJob,
            Path outputPath,
            int minSplitSize,
            int k,
            int p,
            int numReduceTasks,
            Class<? extends Writable> labelClass ) 
    throws ClassNotFoundException, InterruptedException, IOException {
        
        Job job=new Job(conf);
        job.setJobName("Bt-job");
        job.setJarByClass(QJob.class);
        
        
        job.setInputFormatClass(SequenceFileInputFormat.class);
        job.setOutputFormatClass(SequenceFileOutputFormat.class);
        FileInputFormat.setInputPaths(job, inputPathA);
        if ( minSplitSize>0 )
            SequenceFileInputFormat.setMinInputSplitSize(job, minSplitSize);
        FileOutputFormat.setOutputPath(job, outputPath);
        

        
//        MultipleOutputs.addNamedOutput(job, OUTPUT_Bt,
//                SequenceFileOutputFormat.class,
//                QJobKeyWritable.class,QJobValueWritable.class);
        MultipleOutputs.addNamedOutput(job, OUTPUT_Q,
                SequenceFileOutputFormat.class,
                labelClass, 
                VectorWritable.class);
        
        //Warn: tight hadoop integration here:
        job.getConfiguration().set("mapreduce.output.basename", OUTPUT_Bt);
        SequenceFileOutputFormat.setCompressOutput(job, true);
        SequenceFileOutputFormat.setOutputCompressorClass(job, DefaultCodec.class);
        SequenceFileOutputFormat.setOutputCompressionType(job, CompressionType.BLOCK);
        
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(VectorWritable.class);
        
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(VectorWritable.class);
        
        job.setMapperClass(BtMapper.class);
        job.setCombinerClass(OuterProductReducer.class);
        job.setReducerClass(OuterProductReducer.class);
//      job.setPartitionerClass(QPartitioner.class);
        
//        job.getConfiguration().setInt(QJob.PROP_AROWBLOCK_SIZE,aBlockRows );
//        job.getConfiguration().setLong(PROP_OMEGA_SEED, seed);
        job.getConfiguration().setInt(QJob.PROP_K, k);
        job.getConfiguration().setInt(QJob.PROP_P, p);
        job.getConfiguration().set(PROP_QJOB_PATH, inputPathQJob.toString());
        
        // number of reduce tasks doesn't matter. we don't actually 
        // send anything to reducers. in fact, the only reason 
        // we need to configure reduce step is so that combiners can fire.
        // so reduce here is purely symbolic.
        job.setNumReduceTasks(numReduceTasks);
        
        job.submit();
        job.waitForCompletion(false);
        
        if ( !job.isSuccessful())
            throw new IOException ( "Bt job unsuccessful.");
        
        
        
    }
    

}
