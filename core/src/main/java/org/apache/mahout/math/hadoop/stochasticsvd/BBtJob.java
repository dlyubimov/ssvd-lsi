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
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile.CompressionType;
import org.apache.hadoop.io.compress.DefaultCodec;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * Compute B*Bt using simple fact that B*Bt = sum(outer prod ( B_(*i), (B_(*i))
 * ).
 * 
 * @author dmitriy
 * 
 */
public class BBtJob {

    public static final String OUTPUT_BBt = "BBt";

    public static void run(Configuration conf, Path btPath, Path outputPath,
            int numReduceTasks) throws IOException, ClassNotFoundException,
            InterruptedException {

        Job job = new Job(conf);
        job.setJobName("BBt-job");
        // input
        job.setInputFormatClass(SequenceFileInputFormat.class);
        FileInputFormat.setInputPaths(job, btPath);

        // map
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(VectorWritable.class);
        job.setMapperClass(BBtMapper.class);
        job.setReducerClass(BBtReducer.class);

        // combiner and reducer
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(VectorWritable.class);

        // output
        job.setOutputFormatClass(SequenceFileOutputFormat.class);
        FileOutputFormat.setOutputPath(job, outputPath);
        SequenceFileOutputFormat.setOutputCompressionType(job,
                CompressionType.BLOCK);
        SequenceFileOutputFormat.setOutputCompressorClass(job,
                DefaultCodec.class);
        job.getConfiguration().set("mapreduce.output.basename", OUTPUT_BBt);

        // run
        job.submit();
        job.waitForCompletion(false);
        if (!job.isSuccessful())
            throw new IOException("BBt job failed.");
        return;

    }

    // actually, B*Bt matrix is small enough so that we don't need to rely on
    // combiner.
    public static class BBtMapper extends
            Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {

        private VectorWritable m_vw = new VectorWritable();
        private IntWritable m_iw = new IntWritable();
        private UpperTriangular m_bbtPartial; // are all partial BBt products symmetrical as well? yes.

        @Override
        protected void map(IntWritable key, VectorWritable value,
                Context context) throws IOException, InterruptedException {
            Vector btVec = value.get();
            int kp = btVec.size();
            if (m_bbtPartial == null) {
                m_bbtPartial=new UpperTriangular(kp);
            }
            for (int i = 0; i < kp; i++) {
                // this approach should reduce GC churn rate
                double mul = btVec.getQuick(i);
                for (int j = i; j < kp; j++)
                    m_bbtPartial.setQuick(i, j, m_bbtPartial.getQuick(i, j)+mul*btVec.getQuick(j));
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException,
                InterruptedException {
            if ( m_bbtPartial != null ) {
                m_iw.set(context.getTaskAttemptID().getTaskID().getId());
                m_vw.set(new DenseVector(m_bbtPartial.getData(), true));
                context.write(m_iw, m_vw);
            }
            super.cleanup(context);
        }
    }
    
    public static class BBtReducer extends Reducer<IntWritable, VectorWritable, IntWritable, VectorWritable> {

        private double[] m_accum;
        
        @Override
        protected void cleanup(
                Context context)
                throws IOException, InterruptedException {
            try { 
                if ( m_accum != null ) 
                    context.write(new IntWritable(),
                            new VectorWritable(new DenseVector(m_accum,true))
                            );
            } finally {
                super.cleanup(context);
            }
        }

        @Override
        protected void reduce(IntWritable iw, 
                Iterable<VectorWritable> ivw,
                Context ctx)
                throws IOException, InterruptedException {
            Iterator<VectorWritable> vwIter=ivw.iterator();
            Vector bbtPartial=vwIter.next().get();
            if ( m_accum == null ) { 
                m_accum=new double[bbtPartial.size()];
            }
            do { 
                for ( int i = 0; i < m_accum.length;i++) 
                    m_accum[i]+=bbtPartial.getQuick(i);
            } while ( vwIter.hasNext() && null != (bbtPartial=vwIter.next().get()));
        } 
        
    }

    // naive mapper.
//    public static class BBtMapper extends
//            Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {
//
//        private VectorWritable m_vw = new VectorWritable();
//        private IntWritable m_iw = new IntWritable();
//        private DenseVector m_v;
//        double[] m_vRow;
//
//        @Override
//        protected void map(IntWritable key, VectorWritable value,
//                Context context) throws IOException, InterruptedException {
//            Vector btVec = value.get();
//            int kp = btVec.size();
//            if (m_v == null) {
//                m_v = new DenseVector(m_vRow = new double[kp], true);
//                m_vw.set(m_v);
//            }
//            for (int i = 0; i < kp; i++) {
//                // this approach should reduce GC churn rate
//                double mul = btVec.getQuick(i);
//                for (int j = 0; j < kp; j++)
//                    m_vRow[j] = mul * btVec.getQuick(j);
//                m_iw.set(i);
//                context.write(m_iw, m_vw);
//            }
//        }
//    }

}
