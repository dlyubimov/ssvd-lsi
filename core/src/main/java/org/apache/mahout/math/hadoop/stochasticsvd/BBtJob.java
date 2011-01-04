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
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile.CompressionType;
import org.apache.hadoop.io.compress.DefaultCodec;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * Compute B*Bt using simple fact that B*Bt = sum(outer prod ( B_(*i), (B_(*i)) ).
 * 
 * @author dmitriy
 *
 */
public class BBtJob {

    public static final String OUTPUT_BBt = "BBt";
	
    
	public static void run ( Configuration conf, Path btPath, Path outputPath, 
			int numReduceTasks )
	throws IOException, ClassNotFoundException,InterruptedException { 
		
		Job job = new Job(conf);
		job.setJobName("BBt-job");
        job.setJarByClass(BBtJob.class);

	
		// input 
		job.setInputFormatClass(SequenceFileInputFormat.class);
		FileInputFormat.setInputPaths(job, btPath);
		
		// map 
		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(VectorWritable.class);
		job.setMapperClass(BBtMapper.class);
		
		
		// combiner and reducer
		job.setReducerClass(BtJob.OuterProductReducer.class);
		job.setCombinerClass(BtJob.OuterProductReducer.class);
		job.setOutputKeyClass(IntWritable.class);
		job.setOutputValueClass(VectorWritable.class);
		
		// output
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		FileOutputFormat.setOutputPath(job, outputPath);
		SequenceFileOutputFormat.setOutputCompressionType(job, CompressionType.BLOCK);
		SequenceFileOutputFormat.setOutputCompressorClass(job, DefaultCodec.class);
        job.getConfiguration().set("mapreduce.output.basename", OUTPUT_BBt);
		
		// run
		job.submit();
		job.waitForCompletion(false);
		if ( ! job.isSuccessful()) throw new IOException ("BBt job failed.");
		return;
		
	}
	
	public static class BBtMapper extends Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {

		private VectorWritable m_vw = new VectorWritable();
		private IntWritable m_iw = new IntWritable();
		

		@Override
		protected void map(IntWritable key, VectorWritable value,
				Context context) throws IOException, InterruptedException {
			Vector btVec=value.get();
			int kp=btVec.size();
			for ( int i =0; i < kp; i++ ) { 
			    m_iw.set(i);
			    m_vw.set(value.get().times(value.get().getQuick(i)));
			    context.write(m_iw, m_vw);
			}
		}
	}
	
}
