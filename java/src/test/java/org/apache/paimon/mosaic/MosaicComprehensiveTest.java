/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.paimon.mosaic;

import java.io.ByteArrayOutputStream;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.DecimalVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

public class MosaicComprehensiveTest {

    private BufferAllocator allocator;

    @Before
    public void setUp() {
        allocator = new RootAllocator();
    }

    @After
    public void tearDown() {
        allocator.close();
    }

    private byte[] writeToBytes(Schema schema, java.util.function.Consumer<MosaicWriter> writeFn) {
        return writeToBytes(schema, new WriterOptions(), writeFn);
    }

    private byte[] writeToBytes(Schema schema, WriterOptions opts, java.util.function.Consumer<MosaicWriter> writeFn) {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (MosaicWriter writer = new MosaicWriter(baos, schema, opts, allocator)) {
            writeFn.accept(writer);
        }
        return baos.toByteArray();
    }

    private MosaicReader readerFromBytes(byte[] data) {
        InputFile inputFile = (position, buffer, offset, length) -> {
            System.arraycopy(data, (int) position, buffer, offset, length);
        };
        return MosaicReader.open(inputFile, data.length, allocator);
    }

    // Test 1: Large data roundtrip with 1M rows
    @Test
    public void testLargeDataRoundtrip() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("id", new ArrowType.Int(64, true)),
                Field.nullable("name", ArrowType.Utf8.INSTANCE),
                Field.nullable("score", new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)),
                Field.nullable("flag", ArrowType.Bool.INSTANCE)
        ));

        int totalRows = 1_000_000;
        int batchSize = 50_000;

        byte[] data = writeToBytes(arrowSchema, writer -> {
            for (int start = 0; start < totalRows; start += batchSize) {
                int count = Math.min(batchSize, totalRows - start);
                try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
                    BigIntVector ids = (BigIntVector) root.getVector("id");
                    VarCharVector names = (VarCharVector) root.getVector("name");
                    Float8Vector scores = (Float8Vector) root.getVector("score");
                    BitVector flags = (BitVector) root.getVector("flag");

                    ids.allocateNew(count);
                    names.allocateNew(count);
                    scores.allocateNew(count);
                    flags.allocateNew(count);

                    for (int i = 0; i < count; i++) {
                        long val = start + i;
                        ids.set(i, val);
                        names.setSafe(i, ("row_" + val).getBytes());
                        scores.set(i, val * 0.001);
                        flags.set(i, (int) (val % 2));
                    }
                    root.setRowCount(count);
                    writer.write(root);
                }
            }
        });

        assertTrue("File should have data", data.length > 0);

        try (MosaicReader reader = readerFromBytes(data)) {
            int readRows = 0;
            for (int rg = 0; rg < reader.numRowGroups(); rg++) {
                try (VectorSchemaRoot batch = reader.readRowGroup(rg, allocator)) {
                    BigIntVector ids = (BigIntVector) batch.getVector("id");
                    VarCharVector names = (VarCharVector) batch.getVector("name");
                    Float8Vector scores = (Float8Vector) batch.getVector("score");
                    BitVector flags = (BitVector) batch.getVector("flag");

                    for (int i = 0; i < batch.getRowCount(); i++) {
                        long id = ids.get(i);
                        assertEquals("row_" + id, new String(names.get(i)));
                        assertEquals(id * 0.001, scores.get(i), 1e-9);
                        assertEquals((int) (id % 2), flags.get(i));
                    }
                    readRows += batch.getRowCount();
                }
            }
            assertEquals(totalRows, readRows);
        }
    }

    // Test 2: All constant values - should produce small file
    @Test
    public void testAllConstantValues() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("id", new ArrowType.Int(64, true)),
                Field.nullable("name", ArrowType.Utf8.INSTANCE),
                Field.nullable("score", new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)),
                Field.nullable("flag", ArrowType.Bool.INSTANCE)
        ));

        int totalRows = 500_000;
        int batchSize = 50_000;

        byte[] data = writeToBytes(arrowSchema, writer -> {
            for (int start = 0; start < totalRows; start += batchSize) {
                int count = Math.min(batchSize, totalRows - start);
                try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
                    BigIntVector ids = (BigIntVector) root.getVector("id");
                    VarCharVector names = (VarCharVector) root.getVector("name");
                    Float8Vector scores = (Float8Vector) root.getVector("score");
                    BitVector flags = (BitVector) root.getVector("flag");

                    ids.allocateNew(count);
                    names.allocateNew(count);
                    scores.allocateNew(count);
                    flags.allocateNew(count);

                    for (int i = 0; i < count; i++) {
                        ids.set(i, 42L);
                        names.setSafe(i, "constant".getBytes());
                        scores.set(i, 3.14);
                        flags.set(i, 1);
                    }
                    root.setRowCount(count);
                    writer.write(root);
                }
            }
        });

        // Constant data should compress very well
        // A naive uncompressed representation would be at least 500K * (8+8+8+1) = ~12.5MB
        // With compression it should be much smaller
        assertTrue("Constant data file should be very small (got " + data.length + " bytes)",
                data.length < 500_000);

        // Verify data reads back correctly
        try (MosaicReader reader = readerFromBytes(data)) {
            int readRows = 0;
            for (int rg = 0; rg < reader.numRowGroups(); rg++) {
                try (VectorSchemaRoot batch = reader.readRowGroup(rg, allocator)) {
                    for (int i = 0; i < batch.getRowCount(); i++) {
                        assertEquals(42L, ((BigIntVector) batch.getVector("id")).get(i));
                        assertEquals("constant", new String(((VarCharVector) batch.getVector("name")).get(i)));
                        assertEquals(3.14, ((Float8Vector) batch.getVector("score")).get(i), 1e-9);
                        assertEquals(1, ((BitVector) batch.getVector("flag")).get(i));
                    }
                    readRows += batch.getRowCount();
                }
            }
            assertEquals(totalRows, readRows);
        }
    }

    // Test 3: High null rate (95%)
    @Test
    public void testHighNullRate() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("id", new ArrowType.Int(64, true)),
                Field.nullable("name", ArrowType.Utf8.INSTANCE),
                Field.nullable("score", new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE))
        ));

        int totalRows = 500_000;
        int batchSize = 50_000;
        Random rng = new Random(12345);

        byte[] data = writeToBytes(arrowSchema, writer -> {
            Random localRng = new Random(12345);
            for (int start = 0; start < totalRows; start += batchSize) {
                int count = Math.min(batchSize, totalRows - start);
                try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
                    BigIntVector ids = (BigIntVector) root.getVector("id");
                    VarCharVector names = (VarCharVector) root.getVector("name");
                    Float8Vector scores = (Float8Vector) root.getVector("score");

                    ids.allocateNew(count);
                    names.allocateNew(count);
                    scores.allocateNew(count);

                    for (int i = 0; i < count; i++) {
                        long val = start + i;
                        if (localRng.nextDouble() < 0.95) {
                            ids.setNull(i);
                        } else {
                            ids.set(i, val);
                        }
                        if (localRng.nextDouble() < 0.95) {
                            names.setNull(i);
                        } else {
                            names.setSafe(i, ("name_" + val).getBytes());
                        }
                        if (localRng.nextDouble() < 0.95) {
                            scores.setNull(i);
                        } else {
                            scores.set(i, val * 0.5);
                        }
                    }
                    root.setRowCount(count);
                    writer.write(root);
                }
            }
        });

        // Verify roundtrip
        Random verifyRng = new Random(12345);
        try (MosaicReader reader = readerFromBytes(data)) {
            int readRows = 0;
            for (int rg = 0; rg < reader.numRowGroups(); rg++) {
                try (VectorSchemaRoot batch = reader.readRowGroup(rg, allocator)) {
                    BigIntVector ids = (BigIntVector) batch.getVector("id");
                    VarCharVector names = (VarCharVector) batch.getVector("name");
                    Float8Vector scores = (Float8Vector) batch.getVector("score");

                    for (int i = 0; i < batch.getRowCount(); i++) {
                        long val = readRows + i;
                        boolean idNull = verifyRng.nextDouble() < 0.95;
                        boolean nameNull = verifyRng.nextDouble() < 0.95;
                        boolean scoreNull = verifyRng.nextDouble() < 0.95;

                        assertEquals("id null mismatch at row " + val, idNull, ids.isNull(i));
                        if (!idNull) {
                            assertEquals(val, ids.get(i));
                        }
                        assertEquals("name null mismatch at row " + val, nameNull, names.isNull(i));
                        if (!nameNull) {
                            assertEquals("name_" + val, new String(names.get(i)));
                        }
                        assertEquals("score null mismatch at row " + val, scoreNull, scores.isNull(i));
                        if (!scoreNull) {
                            assertEquals(val * 0.5, scores.get(i), 1e-9);
                        }
                    }
                    readRows += batch.getRowCount();
                }
            }
            assertEquals(totalRows, readRows);
        }
    }

    // Test 4: Wide table with 100 columns
    @Test
    public void testWideTable() {
        int numCols = 100;
        int totalRows = 50_000;

        List<Field> fields = new ArrayList<>();
        for (int c = 0; c < numCols; c++) {
            switch (c % 4) {
                case 0:
                    fields.add(Field.nullable("col_" + c, new ArrowType.Int(32, true)));
                    break;
                case 1:
                    fields.add(Field.nullable("col_" + c, new ArrowType.Int(64, true)));
                    break;
                case 2:
                    fields.add(Field.nullable("col_" + c, new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)));
                    break;
                case 3:
                    fields.add(Field.nullable("col_" + c, ArrowType.Utf8.INSTANCE));
                    break;
            }
        }
        Schema arrowSchema = new Schema(fields);

        int batchSize = 10_000;
        byte[] data = writeToBytes(arrowSchema, writer -> {
            for (int start = 0; start < totalRows; start += batchSize) {
                int count = Math.min(batchSize, totalRows - start);
                try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
                    for (int c = 0; c < numCols; c++) {
                        switch (c % 4) {
                            case 0:
                                IntVector iv = (IntVector) root.getVector(c);
                                iv.allocateNew(count);
                                for (int i = 0; i < count; i++) iv.set(i, start + i);
                                break;
                            case 1:
                                BigIntVector bv = (BigIntVector) root.getVector(c);
                                bv.allocateNew(count);
                                for (int i = 0; i < count; i++) bv.set(i, (long)(start + i) * 100);
                                break;
                            case 2:
                                Float8Vector fv = (Float8Vector) root.getVector(c);
                                fv.allocateNew(count);
                                for (int i = 0; i < count; i++) fv.set(i, (start + i) * 0.1);
                                break;
                            case 3:
                                VarCharVector sv = (VarCharVector) root.getVector(c);
                                sv.allocateNew(count);
                                for (int i = 0; i < count; i++) sv.setSafe(i, ("v" + (start + i)).getBytes());
                                break;
                        }
                    }
                    root.setRowCount(count);
                    writer.write(root);
                }
            }
        });

        try (MosaicReader reader = readerFromBytes(data)) {
            assertEquals(numCols, reader.getSchema().getFields().size());
            int readRows = 0;
            for (int rg = 0; rg < reader.numRowGroups(); rg++) {
                try (VectorSchemaRoot batch = reader.readRowGroup(rg, allocator)) {
                    assertEquals(numCols, batch.getFieldVectors().size());
                    readRows += batch.getRowCount();
                }
            }
            assertEquals(totalRows, readRows);
        }
    }

    // Test 5: Many small writes (10000 batches of 10 rows)
    @Test
    public void testManySmallWrites() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("id", new ArrowType.Int(32, true)),
                Field.nullable("val", ArrowType.Utf8.INSTANCE)
        ));

        int numBatches = 10_000;
        int batchSize = 10;
        int totalRows = numBatches * batchSize;

        byte[] data = writeToBytes(arrowSchema, writer -> {
            for (int b = 0; b < numBatches; b++) {
                try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
                    IntVector ids = (IntVector) root.getVector("id");
                    VarCharVector vals = (VarCharVector) root.getVector("val");
                    ids.allocateNew(batchSize);
                    vals.allocateNew(batchSize);
                    for (int i = 0; i < batchSize; i++) {
                        int row = b * batchSize + i;
                        ids.set(i, row);
                        vals.setSafe(i, ("r" + row).getBytes());
                    }
                    root.setRowCount(batchSize);
                    writer.write(root);
                }
            }
        });

        try (MosaicReader reader = readerFromBytes(data)) {
            int readRows = 0;
            for (int rg = 0; rg < reader.numRowGroups(); rg++) {
                try (VectorSchemaRoot batch = reader.readRowGroup(rg, allocator)) {
                    readRows += batch.getRowCount();
                }
            }
            assertEquals(totalRows, readRows);
        }
    }

    // Test 6: Projection with large column count
    @Test
    public void testProjectionWithLargeColumnCount() {
        int numCols = 50;
        List<Field> fields = new ArrayList<>();
        for (int c = 0; c < numCols; c++) {
            fields.add(Field.nullable("col_" + c, new ArrowType.Int(32, true)));
        }
        Schema arrowSchema = new Schema(fields);

        int totalRows = 10_000;
        byte[] data = writeToBytes(arrowSchema, new WriterOptions().numBuckets(2), writer -> {
            try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
                for (int c = 0; c < numCols; c++) {
                    IntVector v = (IntVector) root.getVector(c);
                    v.allocateNew(totalRows);
                    for (int i = 0; i < totalRows; i++) {
                        v.set(i, c * 1000 + i);
                    }
                }
                root.setRowCount(totalRows);
                writer.write(root);
            }
        });

        // Project only 3 out of 50 columns
        try (MosaicReader reader = readerFromBytes(data)) {
            reader.project(new String[]{"col_0", "col_25", "col_49"});
            int readRows = 0;
            for (int rg = 0; rg < reader.numRowGroups(); rg++) {
                try (VectorSchemaRoot batch = reader.readRowGroup(rg, allocator)) {
                    assertEquals(3, batch.getFieldVectors().size());
                    assertEquals("col_0", batch.getVector(0).getName());
                    assertEquals("col_25", batch.getVector(1).getName());
                    assertEquals("col_49", batch.getVector(2).getName());

                    IntVector c0 = (IntVector) batch.getVector(0);
                    IntVector c25 = (IntVector) batch.getVector(1);
                    IntVector c49 = (IntVector) batch.getVector(2);
                    for (int i = 0; i < batch.getRowCount(); i++) {
                        int id = c0.get(i);
                        // id = 0*1000 + original_i, so original_i = id
                        assertEquals(25 * 1000 + id, c25.get(i));
                        assertEquals(49 * 1000 + id, c49.get(i));
                    }
                    readRows += batch.getRowCount();
                }
            }
            assertEquals(totalRows, readRows);
        }
    }

    // Test 7: Sequential vs random data file size comparison
    @Test
    public void testSequentialVsRandomData() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("val", new ArrowType.Int(64, true))
        ));

        int totalRows = 100_000;

        // Sequential data
        byte[] seqData = writeToBytes(arrowSchema, writer -> {
            try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
                BigIntVector vals = (BigIntVector) root.getVector("val");
                vals.allocateNew(totalRows);
                for (int i = 0; i < totalRows; i++) {
                    vals.set(i, (long) i);
                }
                root.setRowCount(totalRows);
                writer.write(root);
            }
        });

        // Random data
        Random rng = new Random(42);
        byte[] randData = writeToBytes(arrowSchema, writer -> {
            try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
                BigIntVector vals = (BigIntVector) root.getVector("val");
                vals.allocateNew(totalRows);
                Random localRng = new Random(42);
                for (int i = 0; i < totalRows; i++) {
                    vals.set(i, localRng.nextLong());
                }
                root.setRowCount(totalRows);
                writer.write(root);
            }
        });

        // Sequential data should compress better than random data
        assertTrue("Sequential data (" + seqData.length + " bytes) should be smaller than random data (" +
                randData.length + " bytes)", seqData.length < randData.length);

        // Verify both roundtrip correctly
        try (MosaicReader reader = readerFromBytes(seqData)) {
            int readRows = 0;
            for (int rg = 0; rg < reader.numRowGroups(); rg++) {
                try (VectorSchemaRoot batch = reader.readRowGroup(rg, allocator)) {
                    readRows += batch.getRowCount();
                }
            }
            assertEquals(totalRows, readRows);
        }

        try (MosaicReader reader = readerFromBytes(randData)) {
            int readRows = 0;
            for (int rg = 0; rg < reader.numRowGroups(); rg++) {
                try (VectorSchemaRoot batch = reader.readRowGroup(rg, allocator)) {
                    readRows += batch.getRowCount();
                }
            }
            assertEquals(totalRows, readRows);
        }
    }

    // Test 8: Multiple row groups roundtrip with small max size
    @Test
    public void testMultipleRowGroupsRoundtrip() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("id", new ArrowType.Int(32, true)),
                Field.nullable("data", new ArrowType.Int(64, true)),
                Field.nullable("text", ArrowType.Utf8.INSTANCE)
        ));

        WriterOptions opts = new WriterOptions()
                .compression(0)
                .numBuckets(1)
                .rowGroupMaxSize(500);

        int totalRows = 1000;
        int batchSize = 50;

        byte[] data = writeToBytes(arrowSchema, opts, writer -> {
            for (int start = 0; start < totalRows; start += batchSize) {
                int count = Math.min(batchSize, totalRows - start);
                try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
                    IntVector ids = (IntVector) root.getVector("id");
                    BigIntVector datas = (BigIntVector) root.getVector("data");
                    VarCharVector texts = (VarCharVector) root.getVector("text");

                    ids.allocateNew(count);
                    datas.allocateNew(count);
                    texts.allocateNew(count);

                    for (int i = 0; i < count; i++) {
                        int val = start + i;
                        ids.set(i, val);
                        datas.set(i, (long) val * 7);
                        texts.setSafe(i, ("item_" + val).getBytes());
                    }
                    root.setRowCount(count);
                    writer.write(root);
                }
            }
        });

        try (MosaicReader reader = readerFromBytes(data)) {
            assertTrue("Should have multiple row groups, got " + reader.numRowGroups(),
                    reader.numRowGroups() > 1);

            int offset = 0;
            for (int rg = 0; rg < reader.numRowGroups(); rg++) {
                try (VectorSchemaRoot batch = reader.readRowGroup(rg, allocator)) {
                    IntVector ids = (IntVector) batch.getVector("id");
                    BigIntVector datas = (BigIntVector) batch.getVector("data");
                    VarCharVector texts = (VarCharVector) batch.getVector("text");

                    for (int i = 0; i < batch.getRowCount(); i++) {
                        int id = ids.get(i);
                        assertEquals((long) id * 7, datas.get(i));
                        assertEquals("item_" + id, new String(texts.get(i)));
                    }
                    offset += batch.getRowCount();
                }
            }
            assertEquals(totalRows, offset);
        }
    }

    // Test 9: Empty string values mixed with non-empty
    @Test
    public void testEmptyStringValues() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("id", new ArrowType.Int(32, true)),
                Field.nullable("text", ArrowType.Utf8.INSTANCE)
        ));

        int totalRows = 100_000;

        byte[] data = writeToBytes(arrowSchema, writer -> {
            try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
                IntVector ids = (IntVector) root.getVector("id");
                VarCharVector texts = (VarCharVector) root.getVector("text");

                ids.allocateNew(totalRows);
                texts.allocateNew(totalRows);

                for (int i = 0; i < totalRows; i++) {
                    ids.set(i, i);
                    if (i % 3 == 0) {
                        texts.setSafe(i, "".getBytes());
                    } else if (i % 3 == 1) {
                        texts.setSafe(i, ("value_" + i).getBytes());
                    } else {
                        texts.setNull(i);
                    }
                }
                root.setRowCount(totalRows);
                writer.write(root);
            }
        });

        try (MosaicReader reader = readerFromBytes(data)) {
            int readRows = 0;
            for (int rg = 0; rg < reader.numRowGroups(); rg++) {
                try (VectorSchemaRoot batch = reader.readRowGroup(rg, allocator)) {
                    IntVector ids = (IntVector) batch.getVector("id");
                    VarCharVector texts = (VarCharVector) batch.getVector("text");

                    for (int i = 0; i < batch.getRowCount(); i++) {
                        int id = ids.get(i);
                        if (id % 3 == 0) {
                            assertFalse(texts.isNull(i));
                            assertEquals("", new String(texts.get(i)));
                        } else if (id % 3 == 1) {
                            assertFalse(texts.isNull(i));
                            assertEquals("value_" + id, new String(texts.get(i)));
                        } else {
                            assertTrue(texts.isNull(i));
                        }
                    }
                    readRows += batch.getRowCount();
                }
            }
            assertEquals(totalRows, readRows);
        }
    }

    // Test 10: Decimal precisions
    @Test
    public void testDecimalPrecisions() {
        // Test Decimal128 with precision 10
        Schema schema10 = new Schema(Arrays.asList(
                Field.nullable("dec10", new ArrowType.Decimal(10, 2, 128))
        ));

        int totalRows = 10_000;
        byte[] data10 = writeToBytes(schema10, writer -> {
            try (VectorSchemaRoot root = VectorSchemaRoot.create(schema10, allocator)) {
                DecimalVector dec = (DecimalVector) root.getVector("dec10");
                dec.allocateNew(totalRows);
                for (int i = 0; i < totalRows; i++) {
                    if (i % 10 == 0) {
                        dec.setNull(i);
                    } else {
                        dec.set(i, new BigDecimal(i + ".99"));
                    }
                }
                root.setRowCount(totalRows);
                writer.write(root);
            }
        });

        try (MosaicReader reader = readerFromBytes(data10)) {
            int readRows = 0;
            for (int rg = 0; rg < reader.numRowGroups(); rg++) {
                try (VectorSchemaRoot batch = reader.readRowGroup(rg, allocator)) {
                    DecimalVector dec = (DecimalVector) batch.getVector("dec10");
                    for (int i = 0; i < batch.getRowCount(); i++) {
                        int id = readRows + i;
                        if (id % 10 == 0) {
                            assertTrue(dec.isNull(i));
                        } else {
                            assertFalse(dec.isNull(i));
                            BigDecimal expected = new BigDecimal(id + ".99");
                            assertEquals(expected, dec.getObject(i));
                        }
                    }
                    readRows += batch.getRowCount();
                }
            }
            assertEquals(totalRows, readRows);
        }

        // Test Decimal128 with precision 38
        Schema schema38 = new Schema(Arrays.asList(
                Field.nullable("dec38", new ArrowType.Decimal(38, 10, 128))
        ));

        byte[] data38 = writeToBytes(schema38, writer -> {
            try (VectorSchemaRoot root = VectorSchemaRoot.create(schema38, allocator)) {
                DecimalVector dec = (DecimalVector) root.getVector("dec38");
                dec.allocateNew(totalRows);
                for (int i = 0; i < totalRows; i++) {
                    if (i % 10 == 0) {
                        dec.setNull(i);
                    } else {
                        // Use a large-precision value
                        BigDecimal val = new BigDecimal("1234567890123456789012345678." + String.format("%010d", i));
                        dec.set(i, val);
                    }
                }
                root.setRowCount(totalRows);
                writer.write(root);
            }
        });

        try (MosaicReader reader = readerFromBytes(data38)) {
            int readRows = 0;
            for (int rg = 0; rg < reader.numRowGroups(); rg++) {
                try (VectorSchemaRoot batch = reader.readRowGroup(rg, allocator)) {
                    DecimalVector dec = (DecimalVector) batch.getVector("dec38");
                    for (int i = 0; i < batch.getRowCount(); i++) {
                        int id = readRows + i;
                        if (id % 10 == 0) {
                            assertTrue(dec.isNull(i));
                        } else {
                            assertFalse(dec.isNull(i));
                            BigDecimal expected = new BigDecimal("1234567890123456789012345678." + String.format("%010d", id));
                            assertEquals(expected, dec.getObject(i));
                        }
                    }
                    readRows += batch.getRowCount();
                }
            }
            assertEquals(totalRows, readRows);
        }
    }
}
