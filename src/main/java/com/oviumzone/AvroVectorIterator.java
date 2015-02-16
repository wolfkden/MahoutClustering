package com.oviumzone;

/**
 * Created by wolf on 2/15/15.
 */

import java.util.*;
import java.io.InputStream;
import java.io.IOException;

import org.apache.avro.file.DataFileStream;
import org.apache.avro.generic.GenericDatumReader;
import org.apache.avro.generic.GenericRecord;
import org.apache.avro.Schema;
import org.apache.avro.Schema.Field;
import org.apache.avro.Schema.Type;
import static org.apache.avro.Schema.Type.BOOLEAN;
import static org.apache.avro.Schema.Type.FLOAT;
import static org.apache.avro.Schema.Type.DOUBLE;
import static org.apache.avro.Schema.Type.INT;
import static org.apache.avro.Schema.Type.LONG;
import static org.apache.avro.Schema.Type.NULL;
import static org.apache.avro.Schema.Type.RECORD;
import static org.apache.avro.Schema.Type.UNION;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import com.google.common.collect.AbstractIterator;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Converts incoming Avro Records into Mahout Vectors.
 * Maps fields to vector positions based on schema order, so if you're going to use this across multiple avro files you MUST specify a reader Schema to get consistent results.
 * Presumes DenseVector currently, eventually will support sparse vector.
 * returned Iterator is not threadsafe.
 * returned Iterator does not support remove.
 * Ignores non-numeric types
 * Maps numeric types to double per java conversion rules.
 * Maps BOOLEAN as TRUE -> 1.0, FALSE -> 0.0.
 * Maps NULL to 0.0.
 */
public class AvroVectorIterator extends AbstractIterator<Vector> {

    private static final Logger LOG = LoggerFactory.getLogger(AvroVectorIterator.class);

    private final DataFileStream<GenericRecord> stream;
    private final List<Field> mappedFields;
    private GenericRecord reuse;

    /**
     * Read Records out of the given InputStream, given a reader Schema
     */
    public AvroVectorIterator(Schema schema, InputStream input) {
        try {
            stream = new DataFileStream(input, new GenericDatumReader(schema));
            final List<Field> fields = schema.getFields();
            List<Field> temp = new ArrayList<Field>(fields.size());
            for (Field field : fields) {
                final Type type = field.schema().getType();
                if (DOUBLE.equals(type) ||
                        FLOAT.equals(type) ||
                        INT.equals(type) ||
                        LONG.equals(type) ||
                        BOOLEAN.equals(type) ||
                        NULL.equals(type)
                        ) {
                    temp.add(field);
                }
            }
            mappedFields = Collections.unmodifiableList(temp);
            LOG.debug("Using {} out of {} potential fields; the rest are non-numeric.", mappedFields.size(), fields.size());
        } catch (IOException exception) {
            throw new RuntimeException(exception);
        }
    }

    /**
     * Read Records out of the given InputStream, using just the writer's Schema.
     */
    public AvroVectorIterator(InputStream input) {
        this(null, input);
    }

    @Override
    protected Vector computeNext() {
        try {
            if (stream.hasNext()) {
                reuse = stream.next();
                Vector vector = new DenseVector(mappedFields.size());
                for (int i = 0; i < mappedFields.size(); i++) {
                    final Field field = mappedFields.get(i);
                    final Object value = reuse.get(field.name());
                    switch(field.schema().getType()) {
                        case DOUBLE:
                        case FLOAT:
                        case INT:
                        case LONG:
                            vector.set(i, ((Number)value).doubleValue());
                            break;
                        case BOOLEAN:
                            vector.set(i, ((Boolean)value).booleanValue() ? 1.0 : 0.0);
                            break;
                        case NULL:
                            vector.set(i, 0.0);
                            break;
                        default:
                            throw new UnsupportedOperationException("Non numeric columns should be igored, but one wasn't.");
                    }
                }
                return vector;
            } else {
                stream.close();
                return endOfData();
            }
        } catch (IOException exception) {
            LOG.error("Underlying Avro library threw an exception.", exception);
            throw new RuntimeException(exception);
        }
    }
}
