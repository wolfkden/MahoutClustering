package com.oviumzone;

/**
 * Created by wolf on 2/15/15.
 */

import java.io.IOException;
import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Iterator;

import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.utils.vectors.io.SequenceFileVectorWriter;

import org.apache.avro.file.DataFileStream;
import org.apache.avro.generic.GenericDatumReader;
import org.apache.avro.generic.GenericRecord;
import org.apache.avro.Schema;
import org.apache.avro.Schema.Field;
import org.apache.avro.Schema.Type;
import static org.apache.avro.Schema.Type.RECORD;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.util.hash.Hash;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.LongWritable;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Goes over a set of source Avro DataFiles in HDFS and writes them into SequenceFiles full of Mahout Vectors.
 * Writes out names of columns that end up in the Vectors.
 * Currently only uses Numeric, BOOLEAN, and NULL types; ignores all others.
 */
public class SequenceFilesFromAvroDataFiles {
    private static final Logger LOG = LoggerFactory.getLogger(SequenceFilesFromAvroDataFiles.class);

    private static final String DISTANCE_HELP = "http://archive.cloudera.com/cdh/3/mahout-0.5-cdh3u4/mahout-core/org/apache/mahout/common/distance/package-summary.html";
    private static final int DEFAULT_SEED_SIZE = 10;

    public static void main(String[] args) throws IOException {

        final Options options = new Options();

        final Option files = new Option("i", "input", true, "An HDFS path glob for the files you want the regression to run on.");
        files.setRequired(true);
        options.addOption(files);

        final Option output = new Option("o", "output", true, "An HDFS path representing a directory to write resultant SequenceFile(s) to; will be created if it doesn't exist.");
        output.setRequired(true);
        options.addOption(output);

        options.addOption("s", "schema", true, "A path to a file containing an avro Schema to use for reading source files.");
        options.addOption("l", "local-schema", false, "flag to indicate schema path is in the local filesystem, rather than hdfs");
        //options.addOption("c", "categories", false, "flag to indicate you would like non-numeric fields handled as categories. WARNING: the set of distinct values must fit in memory.");

        options.addOption("z", "seed", true, "A path to write out an initial random cluster seed.");
        options.addOption("k", "seed-size", true, "Number of samples to use in creating a starting centroid. Defaults to 10.");
        options.addOption("d", "seed-distance-measure", true, "name of a distance measure to use, either FQCN or short name from within Mahout's distance package.\n\t\tSee " + DISTANCE_HELP);

        final CommandLineParser parser = new GnuParser();
        CommandLine line = null;
        try {
            line = parser.parse(options, args);
        } catch (ParseException exception) {
            (new HelpFormatter()).printHelp("hadoop jar mahout-pg-importer-0.0.1-SNAPSHOT.jar", options, true);
            System.exit(-1);
        }

        final Configuration config = new Configuration();
        final FileSystem fs = FileSystem.get(config);

        final ArrayList<Path> paths = new ArrayList<Path>();

        for (FileStatus status : fs.globStatus(new Path(line.getOptionValue("input")))) {
            if (!status.isDir()) {
                paths.add(status.getPath());
            }
        }

        if (1 > paths.size()) {
            throw new IllegalArgumentException("Glob didn't match any files.");
        }

        Schema tempSchema;

        if (line.hasOption("schema")) {
            InputStream input;
            final String path = line.getOptionValue("schema");
            if (line.hasOption("local-schema")) {
                input = new FileInputStream(path);
            } else {
                input = fs.open(new Path(path));
            }
            tempSchema = (new Schema.Parser()).parse(input);
        } else {
            final DataFileStream<GenericRecord> stream = new DataFileStream<GenericRecord>(fs.open(paths.get(0)), new GenericDatumReader<GenericRecord>());
            tempSchema = stream.getSchema();
            stream.close();
        }
        final Schema schema = tempSchema;

        if (!(RECORD.equals(schema.getType()))) {
            throw new UnsupportedOperationException("First retrieved path has to be an avro file with records");
        }

        Path dir = (new Path(line.getOptionValue("output"))).makeQualified(fs);
        if (!fs.mkdirs(dir)) {
            LOG.error("Couldn't create target directory.");
            throw new IOException("Couldn't create target directory.");
        }

    /* XXX warning: this allows the cluster configuration to choose the hashing algorithm, so it may
       not give consistent names when moved between clusters, or if the configuration of our current cluster
       changes.
     */
        final Hash hash = Hash.getInstance(config);
        final Charset UTF8 = Charset.forName("UTF-8");
        for (final Path path : paths) {
            final String uniq = Integer.toHexString(hash.hash(path.getParent().makeQualified(fs).toString().getBytes(UTF8)));
            final Path target = new Path(dir, uniq + "-" + path.getName() + ".seq");
            final InputStream incoming = fs.open(path);
            LOG.info("Converting {} => {}", path, target);
      /* XXX optimization: turn on compression in these files */
            final SequenceFileVectorWriter out = new SequenceFileVectorWriter(SequenceFile.createWriter(fs, config, target, LongWritable.class, VectorWritable.class));
            out.write(new Iterable<Vector>() {
                @Override
                public Iterator<Vector> iterator() {
                    return new AvroVectorIterator(schema, incoming);
                }});
            out.close();
        }
        LOG.info("Wrote {} files to hdfs in {}", paths.size(), dir);
        LOG.info("\tTo list them: `hadoop fs -ls '{}'`", dir);

        if (line.hasOption("seed")) {
            final Path seed = new Path(line.getOptionValue("seed"));
            final int initialClusterSize = line.hasOption("seed-size") ? Integer.parseInt(line.getOptionValue("seed")) : DEFAULT_SEED_SIZE;
            final DistanceMeasure distance = line.hasOption("seed-distance-measure") ? loadDistanceMeasure(line.getOptionValue("seed-distance-measure")) : new EuclideanDistanceMeasure();
            LOG.info("Generating initial cluster using {} samples and distance measure {}", initialClusterSize, distance.getClass().getSimpleName());
            final Path dest = RandomSeedGenerator.buildRandom(config, dir, seed, initialClusterSize, distance);
            LOG.info("Wrote random initial cluster information to {}", dest);
        }
    }

    static DistanceMeasure loadDistanceMeasure(String name) {
        final String error = "Couldn't find a distance measure named '" + name + "'; for some examples see the mahout docs: " + DISTANCE_HELP;
        try {
            Class<? extends DistanceMeasure> clazz = Class.forName(name.contains(".") ? name : "org.apache.mahout.common.distance." + name).asSubclass(DistanceMeasure.class);
            return clazz.newInstance();
        } catch (ClassNotFoundException exception) {
            LOG.error(error);
            throw new RuntimeException("Couldn't find distance class", exception);
        } catch (IllegalAccessException exception) {
            LOG.error(error);
            throw new RuntimeException("Couldn't access distance class", exception);
        } catch (InstantiationException exception) {
            LOG.error(error);
            throw new RuntimeException("Exceptoin while creating a class instance.", exception);
        }
    }
}
