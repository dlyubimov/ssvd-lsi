/* Licensed to the Apache Software Foundation (ASF) under one or more
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

import java.util.Arrays;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.VectorWritable;

/**
 * Mahout CLI adapter for SSVDSolver
 * 
 * @author Dmitriy
 *
 */
public class SSVDCli extends AbstractJob {

    @Override
    public int run(String[] args) throws Exception {
        addInputOption();
        addOutputOption();
        addOption("rank", "k", "decomposition rank", true);
        addOption("oversampling", "p", "oversampling", true);
        addOption("blockHeight", "r", "Y block height (must be > (k+p))", true);
        addOption("minSplitSize", "s", "minimum split size","-1");
        addOption("computeU", "U", "compute U (true/false)", "true");
        addOption("computeV", "V", "compute V (true/false)", "true");
        addOption("reduceTasks", "t", "number of reduce tasks (where applicable)", "1");
        addOption("wide","w", "enable wide input (>30k nonzero elements/row)","false");
        
        Map<String, String> pargs = parseArguments(args);
        String input = pargs.get("--input");
        String output = pargs.get("--output");
        String tempDir = pargs.get("--tempDir");
        int k = Integer.parseInt(pargs.get("--rank"));
        int p = Integer.parseInt(pargs.get ("--oversampling"));
        int r = Integer.parseInt(pargs.get ("--blockHeight"));
        int minSplitSize = Integer.parseInt(pargs.get("--minSplitSize"));
        boolean computeU = Boolean.parseBoolean(pargs.get("--computeU"));
        boolean computeV = Boolean.parseBoolean(pargs.get("--computeV"));
        boolean wide = Boolean.parseBoolean(pargs.get("--wide"));
        int reduceTasks = Integer.parseInt("--reduceTasks");
        
        Configuration conf = getConf();
        SSVDSolver solver = new SSVDSolver(
                conf, 
                new Path[] {new Path(input)}, 
                new Path ( tempDir), 
                r, k, p, reduceTasks, wide);
        solver.setMinSplitSize(minSplitSize);
        solver.setComputeU(computeU);
        solver.setComputeV(computeV);
        
        solver.run();
        
        // housekeeping 
        FileSystem fs = FileSystem.get(conf);
        
        Path outPath = new Path ( output);
        fs.mkdirs(outPath);

        SequenceFile.Writer sigmaW = SequenceFile.createWriter(fs, conf, 
                new Path ( outPath, "sigma"), 
                NullWritable.class, VectorWritable.class);
        try { 
            VectorWritable sValues = new VectorWritable(new DenseVector(
                    Arrays.copyOf(solver.getSingularValues(),k),true));
            sigmaW.append(NullWritable.get(), sValues);
            
        } finally { 
            sigmaW.close();
        }
        
        if ( computeU ) { 
            FileStatus[] uFiles = fs.globStatus(new Path(solver.getUPath()));
            if ( uFiles != null ) for (FileStatus uf:uFiles ) fs.rename(uf.getPath(), outPath);
        }
        if ( computeV ) { 
            FileStatus[] vFiles = fs.globStatus(new Path ( solver.getVPath()));
            if ( vFiles != null ) for ( FileStatus vf:vFiles ) fs.rename(vf.getPath(), outPath);
            
        }
        return 0;
    }

}
