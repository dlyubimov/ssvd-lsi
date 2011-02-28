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
import java.util.List;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.CompressionType;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.common.IOUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.ssvd.EigenSolverWrapper;

/**
 * Stochastic SVD solver (API class).
 * <P>
 * 
 * Implementation details are in my working notes in MAHOUT-376
 * (https://issues.apache.org/jira/browse/MAHOUT-376).
 * <P>
 * 
 * This class is central public API for SSVD solver. The use pattern is as
 * follows:
 * 
 * <UL>
 * <LI>create the solver using constructor and supplying computation parameters.
 * <LI>set optional parameters thru setter methods.
 * <LI>call {@link #run()}.
 * <LI> {@link #getUPath()} (if computed) returns the path to the directory
 * containing m x k U matrix file(s).
 * <LI> {@link #getVPath()} (if computed) returns the path to the directory
 * containing n x k V matrix file(s).
 * 
 * </UL>
 * 
 * 
 * 
 */
public class SSVDSolver {

  private double[] svalues;
  private boolean computeU = true;
  private boolean computeV = true;
  private String uPath;
  private String vPath;

  // configured stuff
  private Configuration conf;
  private Path[] inputPath;
  private Path outputPath;
  private int ablockRows;
  private int k;
  private int p;
  private int reduceTasks;
  private int minSplitSize = -1;
  private boolean cUHalfSigma;
  private boolean cVHalfSigma;

  /**
   * create new SSVD solver. Required parameters are passed to constructor to
   * ensure they are set. Optional parameters can be set using setters .
   * <P>
   * 
   * @param conf
   *          hadoop configuration
   * @param inputPath
   *          Input path (should be compatible with DistributedRowMatrix as of
   *          the time of this writing).
   * @param outputPath
   *          Output path containing U, V and singular values vector files.
   * @param ablockRows
   *          The vertical hight of a q-block (bigger value require more memory
   *          in mappers+ perhaps larger <code>minSplitSize</code> values
   * @param k
   *          desired rank
   * @param p
   *          SSVD oversampling parameter
   * @param reduceTasks
   *          Number of reduce tasks (where applicable)
   * @throws IOException
   *           when IO condition occurs.
   */

  public SSVDSolver(Configuration conf, Path[] inputPath, Path outputPath,
      int ablockRows, int k, int p, int reduceTasks) throws IOException {
    this.conf = conf;
    this.inputPath = inputPath;
    this.outputPath = outputPath;
    this.ablockRows = ablockRows;
    this.k = k;
    this.p = p;
    this.reduceTasks = reduceTasks;
  }

  public void setcUHalfSigma(boolean cUHat) {
    this.cUHalfSigma = cUHat;
  }

  public void setcVHalfSigma(boolean cVHat) {
    this.cVHalfSigma = cVHat;
  }

  /**
   * The setting controlling whether to compute U matrix of low rank SSVD.
   * 
   */
  public void setComputeU(boolean val) {
    computeU = val;
  }

  /**
   * Setting controlling whether to compute V matrix of low-rank SSVD.
   * 
   * @param val
   *          true if we want to output V matrix. Default is true.
   */
  public void setComputeV(boolean val) {
    computeV = val;
  }

  /**
   * Sometimes, if requested A blocks become larger than a split, we may need to
   * use that to ensure at least k+p rows of A get into a split. This is
   * requirement necessary to obtain orthonormalized Q blocks of SSVD.
   * 
   * @param size
   *          the minimum split size to use
   */
  public void setMinSplitSize(int size) {
    minSplitSize = size;
  }

  /**
   * This contains k+p singular values resulted from the solver run.
   * 
   * @return singlular values (largest to smallest)
   */
  public double[] getSingularValues() {
    return svalues;
  }

  /**
   * returns U path (if computation were requested and successful).
   * 
   * @return U output hdfs path, or null if computation was not completed for
   *         whatever reason.
   */
  public String getUPath() {
    return uPath;
  }

  /**
   * return V path ( if computation was requested and successful ) .
   * 
   * @return V output hdfs path, or null if computation was not completed for
   *         whatever reason.
   */
  public String getVPath() {
    return vPath;
  }

  /**
   * run all SSVD jobs.
   * 
   * @throws IOException
   *           if I/O condition occurs.
   */
  public void run() throws IOException {

    LinkedList<Closeable> closeables = new LinkedList<Closeable>();
    try {
      Class<? extends Writable> labelType = sniffInputLabelType(inputPath,
          conf, closeables);
      FileSystem fs = FileSystem.get(conf);

      Path qPath = new Path(outputPath, "Q-job");
      Path btPath = new Path(outputPath, "Bt-job");
      Path bbtPath = new Path(outputPath, "BBt-job");
      Path uHatPath = new Path(outputPath, "UHat");
      Path svPath = new Path(outputPath, "Sigma");
      Path uPath = new Path(outputPath, "U");
      Path vPath = new Path(outputPath, "V");

      fs.delete(qPath, true); // or we can't re-run it repeatedly, just in case.
      fs.delete(btPath, true);
      fs.delete(bbtPath, true);
      fs.delete(uHatPath, true);
      fs.delete(svPath, true);
      fs.delete(uPath, true);
      fs.delete(vPath, true);

      Random rnd = new Random();
      long seed = rnd.nextLong();

      QJob.run(conf, inputPath, qPath, ablockRows, minSplitSize, k, p, seed,
          reduceTasks);

      BtJob.run(conf, inputPath, qPath, btPath, minSplitSize, k, p,
          reduceTasks, labelType);

      BBtJob.run(conf, new Path(btPath, BtJob.OUTPUT_BT + "-*"), bbtPath, 1);

      UpperTriangular bbt = loadUpperTriangularMatrix(fs, new Path(bbtPath,
          BBtJob.OUTPUT_BBT + "-*"), conf);

      // convert bbt to something our eigensolver could understand
      assert bbt.columnSize() == k + p;

      double[][] bbtSquare = new double[k + p][];
      for (int i = 0; i < k + p; i++)
        bbtSquare[i] = new double[k + p];

      for (int i = 0; i < k + p; i++)
        for (int j = i; j < k + p; j++)
          bbtSquare[i][j] = bbtSquare[j][i] = bbt.getQuick(i, j);

      svalues = new double[k + p];

      // try something else.
      EigenSolverWrapper eigenWrapper = new EigenSolverWrapper(bbtSquare);

      double[] eigenva2 = eigenWrapper.getEigenValues();
      for (int i = 0; i < k + p; i++)
        svalues[i] = Math.sqrt(eigenva2[i]); // sqrt?

      // save/redistribute UHat
      //
      double[][] uHat = eigenWrapper.getUHat();

      fs.mkdirs(uHatPath);
      SequenceFile.Writer uHatWriter = SequenceFile.createWriter(fs, conf,
          uHatPath = new Path(uHatPath, "uhat.seq"), IntWritable.class,
          VectorWritable.class, CompressionType.BLOCK);
      closeables.addFirst(uHatWriter);

      int m = uHat.length;
      IntWritable iw = new IntWritable();
      VectorWritable vw = new VectorWritable();
      for (int i = 0; i < m; i++) {
        vw.set(new DenseVector(uHat[i], true));
        iw.set(i);
        uHatWriter.append(iw, vw);
      }

      closeables.remove(uHatWriter);
      uHatWriter.close();

      SequenceFile.Writer svWriter = SequenceFile.createWriter(fs, conf,
          svPath = new Path(svPath, "svalues.seq"), IntWritable.class,
          VectorWritable.class, CompressionType.BLOCK);

      closeables.addFirst(svWriter);

      vw.set(new DenseVector(svalues, true));
      svWriter.append(iw, vw);

      closeables.remove(svWriter);
      svWriter.close();

      UJob ujob = null;
      VJob vjob = null;
      if (computeU)
        (ujob = new UJob()).start(conf,
            new Path(btPath, BtJob.OUTPUT_Q + "-*"), uHatPath, svPath, uPath,
            k, reduceTasks, labelType, cUHalfSigma); // actually this is
                                                     // map-only job anyway

      if (computeV)
        (vjob = new VJob()).start(conf,
            new Path(btPath, BtJob.OUTPUT_BT + "-*"), uHatPath, svPath, vPath,
            k, reduceTasks, cVHalfSigma);

      if (ujob != null) {
        ujob.waitForCompletion();
        this.uPath = uPath.toString();
      }
      if (vjob != null) {
        vjob.waitForCompletion();
        this.vPath = vPath.toString();
      }

    } catch (InterruptedException exc) {
      throw new IOException("Interrupted", exc);
    } catch (ClassNotFoundException exc) {
      throw new IOException(exc);

    } finally {
      IOUtils.closeAll(closeables);
    }

  }

  private static Class<? extends Writable> sniffInputLabelType(
      Path[] inputPath, Configuration conf, LinkedList<Closeable> closeables)
    throws IOException {
    FileSystem fs = FileSystem.get(conf);
    for (Path p : inputPath) {
      FileStatus[] fstats = fs.globStatus(p);
      if (fstats == null || fstats.length == 0)
        continue;
      SequenceFile.Reader r = new SequenceFile.Reader(fs, fstats[0].getPath(),
          conf);
      closeables.addFirst(r);

      try {
        return r.getKeyClass().asSubclass(Writable.class);
      } finally {
        closeables.remove(r);
        r.close();
      }
    }

    throw new IOException(
        "Unable to open input files to determine input label type.");
  }

  private static final Pattern OUTPUT_FILE_PATTERN = Pattern
      .compile("(\\w+)-(m|r)-(\\d+)(\\.\\w+)?");

  static Comparator<FileStatus> partitionComparator = new Comparator<FileStatus>() {
    private Matcher matcher = OUTPUT_FILE_PATTERN.matcher("");

    @Override
    public int compare(FileStatus o1, FileStatus o2) {
      matcher.reset(o1.getPath().getName());
      if (!matcher.matches())
        throw new RuntimeException(String.format(
            "Unexpected file name, unable to deduce partition #:%s", o1
                .getPath().toString()));
      int p1 = Integer.parseInt(matcher.group(3));
      matcher.reset(o2.getPath().getName());
      if (!matcher.matches())
        throw new RuntimeException(String.format(
            "Unexpected file name, unable to deduce partition #:%s", o2
                .getPath().toString()));

      int p2 = Integer.parseInt(matcher.group(3));
      return p1 - p2;
    }

  };

  /**
   * helper capabiltiy to load distributed row matrices into dense matrix (to
   * support tests mainly).
   * 
   * @param fs
   *          filesystem
   * @param glob
   *          FS glob
   * @param conf
   *          configuration
   * @return Dense matrix array
   * @throws IOException
   *           when I/O occurs.
   */
  public static double[][] loadDistributedRowMatrix(FileSystem fs, Path glob,
      Configuration conf) throws IOException {

    FileStatus[] files = fs.globStatus(glob);
    if (files == null)
      return null;

    List<double[]> denseData = new ArrayList<double[]>();
    IntWritable iw = new IntWritable();
    VectorWritable vw = new VectorWritable();

    // int m=0;

    // assume it is partitioned output, so we need to read them up
    // in order of partitions.
    Arrays.sort(files, partitionComparator);

    for (FileStatus fstat : files) {
      Path file = fstat.getPath();
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, file, conf);
      try {
        while (reader.next(iw, vw)) {
          Vector v = vw.get();
          int size;
          double[] row = new double[size = v.size()];
          for (int i = 0; i < size; i++)
            row[i] = v.get(i);
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

  public static UpperTriangular loadUpperTriangularMatrix(FileSystem fs,
      Path glob, Configuration conf) throws IOException {

    FileStatus[] files = fs.globStatus(glob);
    if (files == null)
      return null;

    IntWritable iw = new IntWritable();
    VectorWritable vw = new VectorWritable();
    UpperTriangular result = null;

    // assume it is partitioned output, so we need to read them up
    // in order of partitions.
    Arrays.sort(files, partitionComparator);

    for (FileStatus fstat : files) {
      Path file = fstat.getPath();
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, file, conf);
      try {
        while (reader.next(iw, vw)) {
          Vector v = vw.get();
          if (result == null)
            result = new UpperTriangular(v);
          else
            throw new IOException(
                "Unexpected overrun in upper triangular matrix files");
        }
      } finally {
        reader.close();
      }
    }

    if (result == null)
      throw new IOException(
          "Unexpected underrun in upper triangular matrix files");
    return result;
  }

}