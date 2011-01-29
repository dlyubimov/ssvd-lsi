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

package org.apache.mahout.utils.vectors;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.cli2.util.HelpFormatter;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.Charset;

/**
 * Can read in a {@link SequenceFile} of {@link Vector}s and dump
 * out the results using {@link Vector#asFormatString()} to either the console or to a
 * file.
 */
public final class VectorDumper {

  private static final Logger log = LoggerFactory.getLogger(VectorDumper.class);

  private VectorDumper() {
  }

  public static void main(String[] args) throws Exception {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option seqOpt = obuilder.withLongName("seqFile").withRequired(false).withArgument(
            abuilder.withName("seqFile").withMinimum(1).withMaximum(1).create()).withDescription(
            "The Sequence File containing the Vectors").withShortName("s").create();
    Option vectorAsKeyOpt = obuilder.withLongName("useKey").withRequired(false).withDescription(
            "If the Key is a vector, then dump that instead").withShortName("u").create();
    Option printKeyOpt = obuilder.withLongName("printKey").withRequired(false).withDescription(
            "Print out the key as well, delimited by a tab (or the value if useKey is true)").withShortName("p")
            .create();
    Option outputOpt = obuilder.withLongName("output").withRequired(false).withArgument(
            abuilder.withName("output").withMinimum(1).withMaximum(1).create()).withDescription(
            "The output file.  If not specified, dumps to the console").withShortName("o").create();
    Option dictOpt = obuilder.withLongName("dictionary").withRequired(false).withArgument(
            abuilder.withName("dictionary").withMinimum(1).withMaximum(1).create()).withDescription(
            "The dictionary file. ").withShortName("d").create();
    Option dictTypeOpt = obuilder.withLongName("dictionaryType").withRequired(false).withArgument(
            abuilder.withName("dictionaryType").withMinimum(1).withMaximum(1).create()).withDescription(
            "The dictionary file type (text|sequencefile)").withShortName("dt").create();
    Option centroidJSonOpt = obuilder.withLongName("json").withRequired(false).withDescription(
            "Output the centroid as JSON.  Otherwise it substitutes in the terms for vector cell entries")
            .withShortName("j").create();
    Option sizeOpt = obuilder.withLongName("sizeOnly").withRequired(false).
            withDescription("Dump only the size of the vector").withShortName("sz").create();
    Option helpOpt = obuilder.withLongName("help").withDescription("Print out help").withShortName("h")
            .create();

    Group group = gbuilder.withName("Options").withOption(seqOpt).withOption(outputOpt).withOption(
            dictTypeOpt).withOption(dictOpt).withOption(centroidJSonOpt).withOption(vectorAsKeyOpt).withOption(
            printKeyOpt).withOption(sizeOpt).create();

    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);

      if (cmdLine.hasOption(helpOpt)) {

        printHelp(group);
        return;
      }

      if (cmdLine.hasOption(seqOpt)) {
        Path path = new Path(cmdLine.getValue(seqOpt).toString());
        //System.out.println("Input Path: " + path); interferes with output?
        Configuration conf = new Configuration();

        FileSystem fs = FileSystem.get(path.toUri(), conf);

        String dictionaryType = "text";
        if (cmdLine.hasOption(dictTypeOpt)) {
          dictionaryType = cmdLine.getValue(dictTypeOpt).toString();
        }

        String[] dictionary = null;
        if (cmdLine.hasOption(dictOpt)) {
          if ("text".equals(dictionaryType)) {
            dictionary = VectorHelper.loadTermDictionary(new File(cmdLine.getValue(dictOpt).toString()));
          } else if ("sequencefile".equals(dictionaryType)) {
            dictionary = VectorHelper.loadTermDictionary(conf, fs, cmdLine.getValue(dictOpt).toString());
          } else {
            throw new OptionException(dictTypeOpt);
          }
        }
        boolean useJSON = cmdLine.hasOption(centroidJSonOpt);
        boolean sizeOnly = cmdLine.hasOption(sizeOpt);
        SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);

        Writable keyWritable = reader.getKeyClass().asSubclass(Writable.class).newInstance();
        Writable valueWritable = reader.getValueClass().asSubclass(Writable.class).newInstance();
        boolean transposeKeyValue = cmdLine.hasOption(vectorAsKeyOpt);
        try {
          Writer writer;
          if (cmdLine.hasOption(outputOpt)) {
            writer = new OutputStreamWriter(
                new FileOutputStream(new File(cmdLine.getValue(outputOpt).toString())), Charset.forName("UTF-8"));
          } else {
            writer = new OutputStreamWriter(System.out);
          }
          try {
            boolean printKey = cmdLine.hasOption(printKeyOpt);
            long i = 0;
            while (reader.next(keyWritable, valueWritable)) {
              if (printKey) {
                Writable notTheVectorWritable = transposeKeyValue ? valueWritable : keyWritable;
                writer.write(notTheVectorWritable.toString());
                writer.write('\t');
              }
              VectorWritable vectorWritable = (VectorWritable) (transposeKeyValue ? keyWritable : valueWritable);
              Vector vector = vectorWritable.get();
              if (sizeOnly) {
                if (vector instanceof NamedVector) {
                  writer.write(((NamedVector) vector).getName());
                  writer.write(":");
                } else {
                  writer.write(String.valueOf(i++));
                  writer.write(":");
                }
                writer.write(String.valueOf(vector.size()));
                writer.write('\n');
              } else {
                String fmtStr = useJSON ? vector.asFormatString() : VectorHelper.vectorToString(vector, dictionary);
                writer.write(fmtStr);
                writer.write('\n');
              }
            }
          } finally {
            writer.close();
          }
        } finally {
          reader.close();
        }
      }

    } catch (OptionException e) {
      log.error("Exception", e);
      printHelp(group);
    }

  }

  private static void printHelp(Group group) {
    HelpFormatter formatter = new HelpFormatter();
    formatter.setGroup(group);
    formatter.print();
  }
}
