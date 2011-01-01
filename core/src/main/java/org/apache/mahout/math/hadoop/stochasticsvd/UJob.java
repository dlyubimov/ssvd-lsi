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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile.CompressionType;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.compress.DefaultCodec;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * Computes U=Q*Uhat of SSVD
 * 
 * @author Dmitriy
 *
 */
public class UJob {
    private static final String     OUTPUT_U = "u";
    private static final String     PROP_UHAT_PATH = "ssvd.uhat.path";
    private static final String     PROP_K = "ssvd.k";
    
    private Job         m_job;
    
    public void start ( Configuration conf, 
            Path inputPathQ,
            Path inputUHatPath,
            Path outputPath,
            int k,
            int numReduceTasks, 
            Class<?extends Writable> labelClass ) 
    throws ClassNotFoundException, InterruptedException, IOException {
        
        m_job=new Job(conf);
        m_job.setJobName("U-job");
        m_job.setJarByClass(UJob.class);
        
        
        m_job.setInputFormatClass(SequenceFileInputFormat.class);
        m_job.setOutputFormatClass(SequenceFileOutputFormat.class);
        FileInputFormat.setInputPaths(m_job, inputPathQ);
        FileOutputFormat.setOutputPath(m_job, outputPath);
        

        //Warn: tight hadoop integration here:
        m_job.getConfiguration().set("mapreduce.output.basename", OUTPUT_U);
        SequenceFileOutputFormat.setCompressOutput(m_job, true);
        SequenceFileOutputFormat.setOutputCompressorClass(m_job, DefaultCodec.class);
        SequenceFileOutputFormat.setOutputCompressionType(m_job, CompressionType.BLOCK);
        
        m_job.setMapOutputKeyClass(IntWritable.class);
        m_job.setMapOutputValueClass(VectorWritable.class);
        
        m_job.setOutputKeyClass(labelClass);
        m_job.setOutputValueClass(VectorWritable.class);
                
        m_job.getConfiguration().set(PROP_UHAT_PATH, inputUHatPath.toString());
        m_job.getConfiguration().setInt(PROP_K, k);
        m_job.setNumReduceTasks(0);
        m_job.submit();
        
    }
    
    public void waitForCompletion () throws IOException, ClassNotFoundException, InterruptedException{ 
        m_job.waitForCompletion(false);
        
        if ( !m_job.isSuccessful())
            throw new IOException ( "U job unsuccessful.");
        
    }
    
    public static final class UMapper extends Mapper<Writable, VectorWritable, Writable, VectorWritable> {

        private Matrix m_uHat;
        private DenseVector    m_uRow;
        private VectorWritable m_uRowWritable;
        private int             m_kp;
        private int             m_k;
        
        @Override
        protected void map(Writable key, VectorWritable value,
                Context context) throws IOException, InterruptedException {
            Vector qRow = value.get();
            for ( int i = 0; i < m_k; i++ ) 
                m_uRow.setQuick(i, qRow.dot(m_uHat.getColumn(i)));
            context.write(key, m_uRowWritable); // U inherits original A row labels.
        }

        @Override
        protected void setup(Context context) throws IOException,
                InterruptedException {
            super.setup(context);
            FileSystem fs = FileSystem.get(context.getConfiguration());
            Path uHatPath = new Path ( context.getConfiguration().get(PROP_UHAT_PATH));
            m_uHat = new DenseMatrix ( SSVDSolver.loadDistributedRowMatrix(fs, uHatPath, context.getConfiguration()));
            // since uHat is (k+p) x (k+p)
            m_kp = m_uHat.columnSize();
            m_k=context.getConfiguration().getInt(PROP_K,m_kp);
            m_uRow = new DenseVector ( m_k);
            m_uRowWritable = new VectorWritable(m_uRow);
            
        } 
        
    }

}
