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

package org.apache.mahout.utils.vectors.io;

import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Collection;

import junit.framework.Assert;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.utils.vectors.RandomVectorIterable;

public class VectorWriterTest extends MahoutTestCase {

  public void testSFVW() throws Exception {
    Path path = getTestTempFilePath("sfvw");
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    SequenceFile.Writer seqWriter = new SequenceFile.Writer(fs, conf, path, LongWritable.class, VectorWritable.class);
    SequenceFileVectorWriter writer = new SequenceFileVectorWriter(seqWriter);
    Iterable<Vector> iter = new RandomVectorIterable(50);
    writer.write(iter);
    writer.close();
    
    SequenceFile.Reader seqReader = new SequenceFile.Reader(fs, path, conf);
    Writable key = new LongWritable();
    Writable value = new VectorWritable();
    int count = 0;
    while (seqReader.next(key, value)){
      count++;
    }
    Assert.assertEquals(count + " does not equal: " + 50, 50, count);
  }
  
  public void test() throws Exception {
    StringWriter strWriter = new StringWriter();
    VectorWriter writer = new JWriterVectorWriter(strWriter);
    Collection<Vector> vectors = new ArrayList<Vector>();
    vectors.add(new DenseVector(new double[]{0.3, 1.5, 4.5}));
    vectors.add(new DenseVector(new double[]{1.3, 1.5, 3.5}));
    writer.write(vectors);
    writer.close();
    String buffer = strWriter.toString();
    Assert.assertNotNull(buffer);
    Assert.assertTrue(buffer.length() > 0);
    
  }
}
