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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.MatrixWritable;

/**
 * Ad-hoc substitution for {@link MatrixWritable}. Perhaps more useful for
 * situations with mostly dense data (such as Q-blocks) but reduces GC by
 * reusing the same block memory between loads and writes.
 * <p>
 * 
 * in case of Q blocks, it doesn't even matter if they this data is dense cause
 * we need to unpack it into dense for fast access in computations anyway and
 * even if it is not so dense the block compressor in sequence files will take
 * care of it for the serialized size.
 * <P>
 * 
 * @author Dmitriy
 * 
 */
public class DenseBlockWritable implements Writable {
  double[][] m_block;

  public DenseBlockWritable() {
    super();
  }

  public void setBlock(double[][] block) {
    m_block = block;
  }

  public double[][] getBlock() {
    return m_block;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    int m = in.readInt();
    int n = in.readInt();
    if (m_block == null)
      m_block = new double[m][0];
    else if (m_block.length != m)
      m_block = Arrays.copyOf(m_block, m);
    for (int i = 0; i < m; i++) {
      if (m_block[i] == null || m_block[i].length != n)
        m_block[i] = new double[n];
      for (int j = 0; j < n; j++)
        m_block[i][j] = in.readDouble();

    }
  }

  @Override
  public void write(DataOutput out) throws IOException {
    int m = m_block.length;
    int n = m_block.length == 0 ? 0 : m_block[0].length;

    out.writeInt(m);
    out.writeInt(n);
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
        out.writeDouble(m_block[i][j]);
  }

}
