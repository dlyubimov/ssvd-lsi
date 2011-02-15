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
public class VJob {
  private static final String OUTPUT_V = "v";
  private static final String PROP_UHat_PATH = "ssvd.uhat.path";
  private static final String PROP_SIGMA_PATH = "ssvd.sigma.path";
  private static final String PROP_V_HALFSIGMA = "ssvd.v.halfsigma";
  private static final String PROP_K = "ssvd.k";

  private Job m_job;

  public void start(Configuration conf, Path inputPathBt, Path inputUHatPath,
      Path inputSigmaPath, Path outputPath, int k, int numReduceTasks,
      boolean vHalfSigma) throws ClassNotFoundException, InterruptedException,
      IOException {

    m_job = new Job(conf);
    m_job.setJobName("V-job");
    m_job.setJarByClass(VJob.class);

    m_job.setInputFormatClass(SequenceFileInputFormat.class);
    m_job.setOutputFormatClass(SequenceFileOutputFormat.class);
    FileInputFormat.setInputPaths(m_job, inputPathBt);
    FileOutputFormat.setOutputPath(m_job, outputPath);

    // Warn: tight hadoop integration here:
    m_job.getConfiguration().set("mapreduce.output.basename", OUTPUT_V);
    SequenceFileOutputFormat.setCompressOutput(m_job, true);
    SequenceFileOutputFormat
        .setOutputCompressorClass(m_job, DefaultCodec.class);
    SequenceFileOutputFormat.setOutputCompressionType(m_job,
        CompressionType.BLOCK);

    m_job.setMapOutputKeyClass(IntWritable.class);
    m_job.setMapOutputValueClass(VectorWritable.class);

    m_job.setOutputKeyClass(IntWritable.class);
    m_job.setOutputValueClass(VectorWritable.class);

    m_job.setMapperClass(VMapper.class);

    m_job.getConfiguration().set(PROP_UHat_PATH, inputUHatPath.toString());
    m_job.getConfiguration().set(PROP_SIGMA_PATH, inputSigmaPath.toString());
    if (vHalfSigma)
      m_job.getConfiguration().set(PROP_V_HALFSIGMA, "y");
    m_job.getConfiguration().setInt(PROP_K, k);
    m_job.setNumReduceTasks(0);
    m_job.submit();

  }

  public void waitForCompletion() throws IOException, ClassNotFoundException,
      InterruptedException {
    m_job.waitForCompletion(false);

    if (!m_job.isSuccessful())
      throw new IOException("V job unsuccessful.");

  }

  public static final class VMapper extends
      Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {

    private Matrix m_uHat;
    private DenseVector m_vRow;
    private DenseVector m_sValues;
    private VectorWritable m_vRowWritable;
    private int m_kp;
    private int m_k;

    @Override
    protected void map(IntWritable key, VectorWritable value, Context context)
        throws IOException, InterruptedException {
      Vector qRow = value.get();
      for (int i = 0; i < m_k; i++)
        m_vRow.setQuick(i,
            qRow.dot(m_uHat.getColumn(i)) / m_sValues.getQuick(i));
      context.write(key, m_vRowWritable); // U inherits original A row labels.
    }

    @Override
    protected void setup(Context context) throws IOException,
        InterruptedException {
      super.setup(context);
      FileSystem fs = FileSystem.get(context.getConfiguration());
      Path uHatPath = new Path(context.getConfiguration().get(PROP_UHat_PATH));

      Path sigmaPath = new Path(context.getConfiguration().get(PROP_SIGMA_PATH));

      m_uHat = new DenseMatrix(SSVDSolver.loadDistributedRowMatrix(fs,
          uHatPath, context.getConfiguration()));
      // since uHat is (k+p) x (k+p)
      m_kp = m_uHat.columnSize();
      m_k = context.getConfiguration().getInt(PROP_K, m_kp);
      m_vRow = new DenseVector(m_k);
      m_vRowWritable = new VectorWritable(m_vRow);

      m_sValues = new DenseVector(SSVDSolver.loadDistributedRowMatrix(fs,
          sigmaPath, context.getConfiguration())[0], true);
      if (context.getConfiguration().get(PROP_V_HALFSIGMA) != null)
        for (int i = 0; i < m_k; i++)
          m_sValues.setQuick(i, Math.sqrt(m_sValues.getQuick(i)));

    }

  }

}
