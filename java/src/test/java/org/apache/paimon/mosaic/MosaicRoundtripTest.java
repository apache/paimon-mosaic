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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.List;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.SmallIntVector;
import org.apache.arrow.vector.TinyIntVector;
import org.apache.arrow.vector.VarBinaryVector;
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

public class MosaicRoundtripTest {

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

    @Test
    public void testBasicRoundtrip() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.notNullable("id", new ArrowType.Int(32, true)),
                Field.nullable("name", ArrowType.Utf8.INSTANCE),
                Field.nullable("score", new ArrowType.FloatingPoint(org.apache.arrow.vector.types.FloatingPointPrecision.DOUBLE))
        ));

        byte[] data;
        try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
            IntVector ids = (IntVector) root.getVector("id");
            VarCharVector names = (VarCharVector) root.getVector("name");
            Float8Vector scores = (Float8Vector) root.getVector("score");

            ids.allocateNew(50);
            names.allocateNew(50);
            scores.allocateNew(50);

            for (int i = 0; i < 50; i++) {
                ids.set(i, i);
                names.setSafe(i, ("user_" + i).getBytes());
                scores.set(i, i * 1.5);
            }
            root.setRowCount(50);

            data = writeToBytes(arrowSchema, new WriterOptions().numBuckets(2), writer -> writer.write(root));
        }

        assertTrue(data.length > 32);
        assertEquals('M', data[data.length - 4]);
        assertEquals('O', data[data.length - 3]);
        assertEquals('S', data[data.length - 2]);
        assertEquals('A', data[data.length - 1]);

        try (MosaicReader reader = readerFromBytes(data)) {
            assertEquals(3, reader.getSchema().getFields().size());
            assertTrue(reader.numRowGroups() >= 1);

            int idCol = reader.getSchema().getFields().indexOf(reader.getSchema().findField("id"));
            int nameCol = reader.getSchema().getFields().indexOf(reader.getSchema().findField("name"));
            int scoreCol = reader.getSchema().getFields().indexOf(reader.getSchema().findField("score"));
            assertTrue(idCol >= 0);
            assertTrue(nameCol >= 0);
            assertTrue(scoreCol >= 0);

            int totalRows = 0;
            for (int rg = 0; rg < reader.numRowGroups(); rg++) {
                try (VectorSchemaRoot batch = reader.readRowGroup(rg, allocator)) {
                    int rows = batch.getRowCount();
                    totalRows += rows;

                    IntVector readIds = (IntVector) batch.getVector(idCol);
                    VarCharVector readNames = (VarCharVector) batch.getVector(nameCol);
                    Float8Vector readScores = (Float8Vector) batch.getVector(scoreCol);

                    for (int i = 0; i < rows; i++) {
                        int id = readIds.get(i);
                        String name = new String(readNames.get(i));
                        double score = readScores.get(i);
                        assertEquals("user_" + id, name);
                        assertEquals(id * 1.5, score, 1e-9);
                    }
                }
            }
            assertEquals(50, totalRows);
        }
    }

    @Test
    public void testNullValues() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("id", new ArrowType.Int(32, true)),
                Field.nullable("name", ArrowType.Utf8.INSTANCE),
                Field.nullable("value", new ArrowType.FloatingPoint(org.apache.arrow.vector.types.FloatingPointPrecision.DOUBLE))
        ));

        byte[] data;
        try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
            IntVector ids = (IntVector) root.getVector("id");
            VarCharVector names = (VarCharVector) root.getVector("name");
            Float8Vector values = (Float8Vector) root.getVector("value");

            ids.allocateNew(3);
            names.allocateNew(3);
            values.allocateNew(3);

            ids.set(0, 1);
            names.setSafe(0, "hello".getBytes());
            values.set(0, 1.0);

            ids.set(1, 2);
            names.setNull(1);
            values.set(1, 2.0);

            ids.set(2, 3);
            names.setSafe(2, "world".getBytes());
            values.setNull(2);

            root.setRowCount(3);
            data = writeToBytes(arrowSchema, writer -> writer.write(root));
        }

        try (MosaicReader reader = readerFromBytes(data)) {
            int nameCol = reader.getSchema().getFields().indexOf(reader.getSchema().findField("name"));
            int valueCol = reader.getSchema().getFields().indexOf(reader.getSchema().findField("value"));

            for (int rg = 0; rg < reader.numRowGroups(); rg++) {
                try (VectorSchemaRoot batch = reader.readRowGroup(rg, allocator)) {
                    assertEquals(3, batch.getRowCount());

                    VarCharVector readNames = (VarCharVector) batch.getVector(nameCol);
                    Float8Vector readValues = (Float8Vector) batch.getVector(valueCol);

                    assertFalse(readNames.isNull(0));
                    assertEquals("hello", new String(readNames.get(0)));

                    assertTrue(readNames.isNull(1));

                    assertFalse(readNames.isNull(2));
                    assertEquals("world", new String(readNames.get(2)));

                    assertFalse(readValues.isNull(0));
                    assertTrue(readValues.isNull(2));
                }
            }
        }
    }

    @Test
    public void testProjection() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("a", new ArrowType.Int(32, true)),
                Field.nullable("b", ArrowType.Utf8.INSTANCE),
                Field.nullable("c", new ArrowType.FloatingPoint(org.apache.arrow.vector.types.FloatingPointPrecision.DOUBLE)),
                Field.nullable("d", ArrowType.Utf8.INSTANCE)
        ));

        byte[] data;
        try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
            IntVector aVec = (IntVector) root.getVector("a");
            VarCharVector bVec = (VarCharVector) root.getVector("b");
            Float8Vector cVec = (Float8Vector) root.getVector("c");
            VarCharVector dVec = (VarCharVector) root.getVector("d");

            int n = 20;
            aVec.allocateNew(n);
            bVec.allocateNew(n);
            cVec.allocateNew(n);
            dVec.allocateNew(n);

            for (int i = 0; i < n; i++) {
                aVec.set(i, i);
                bVec.setSafe(i, ("val_" + i).getBytes());
                cVec.set(i, (double) i);
                dVec.setSafe(i, ("extra_" + i).getBytes());
            }
            root.setRowCount(n);
            data = writeToBytes(arrowSchema, new WriterOptions().numBuckets(2), writer -> writer.write(root));
        }

        try (MosaicReader reader = readerFromBytes(data)) {
            int aCol = reader.getSchema().getFields().indexOf(reader.getSchema().findField("a"));
            int bCol = reader.getSchema().getFields().indexOf(reader.getSchema().findField("b"));
            int[] projected = {aCol, bCol};

            int totalRows = 0;
            for (int rg = 0; rg < reader.numRowGroups(); rg++) {
                try (VectorSchemaRoot batch = reader.readRowGroup(rg, projected, allocator)) {
                    totalRows += batch.getRowCount();
                    assertEquals(2, batch.getFieldVectors().size());
                }
            }
            assertEquals(20, totalRows);
        }
    }

    @Test
    public void testStats() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("id", new ArrowType.Int(32, true)),
                Field.nullable("name", ArrowType.Utf8.INSTANCE),
                Field.nullable("value", new ArrowType.FloatingPoint(org.apache.arrow.vector.types.FloatingPointPrecision.DOUBLE))
        ));

        WriterOptions opts = new WriterOptions().statsColumns(0, 2);

        byte[] data;
        try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
            IntVector ids = (IntVector) root.getVector("id");
            VarCharVector names = (VarCharVector) root.getVector("name");
            Float8Vector values = (Float8Vector) root.getVector("value");

            int n = 10;
            ids.allocateNew(n);
            names.allocateNew(n);
            values.allocateNew(n);

            for (int i = 0; i < n; i++) {
                ids.set(i, i * 10);
                names.setSafe(i, ("item_" + i).getBytes());
                values.set(i, i * 1.1);
            }
            root.setRowCount(n);
            data = writeToBytes(arrowSchema, opts, writer -> writer.write(root));
        }

        try (MosaicReader reader = readerFromBytes(data)) {
            for (int rg = 0; rg < reader.numRowGroups(); rg++) {
                java.util.List<ColumnStatistics> stats = reader.getRowGroupStatistics(rg);
                assertTrue(stats.size() > 0);
                for (ColumnStatistics stat : stats) {
                    assertTrue(stat.getColumnIndex() == 0 || stat.getColumnIndex() == 2);
                    assertEquals(0, stat.getNullCount());
                    assertTrue(stat.hasMinMax());
                    assertNotNull(stat.getMin());
                    assertNotNull(stat.getMax());
                    assertTrue(stat.getMin().length > 0);
                    assertTrue(stat.getMax().length > 0);
                }
            }
        }
    }

    @Test
    public void testAllTypes() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("f_bool", ArrowType.Bool.INSTANCE),
                Field.nullable("f_int8", new ArrowType.Int(8, true)),
                Field.nullable("f_int16", new ArrowType.Int(16, true)),
                Field.nullable("f_int32", new ArrowType.Int(32, true)),
                Field.nullable("f_int64", new ArrowType.Int(64, true)),
                Field.nullable("f_float32", new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)),
                Field.nullable("f_float64", new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)),
                Field.nullable("f_utf8", ArrowType.Utf8.INSTANCE),
                Field.nullable("f_binary", ArrowType.Binary.INSTANCE)
        ));

        byte[] data;
        try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
            BitVector boolVec = (BitVector) root.getVector("f_bool");
            TinyIntVector int8Vec = (TinyIntVector) root.getVector("f_int8");
            SmallIntVector int16Vec = (SmallIntVector) root.getVector("f_int16");
            IntVector int32Vec = (IntVector) root.getVector("f_int32");
            BigIntVector int64Vec = (BigIntVector) root.getVector("f_int64");
            Float4Vector f32Vec = (Float4Vector) root.getVector("f_float32");
            Float8Vector f64Vec = (Float8Vector) root.getVector("f_float64");
            VarCharVector utf8Vec = (VarCharVector) root.getVector("f_utf8");
            VarBinaryVector binVec = (VarBinaryVector) root.getVector("f_binary");

            int n = 2;
            boolVec.allocateNew(n);
            int8Vec.allocateNew(n);
            int16Vec.allocateNew(n);
            int32Vec.allocateNew(n);
            int64Vec.allocateNew(n);
            f32Vec.allocateNew(n);
            f64Vec.allocateNew(n);
            utf8Vec.allocateNew(n);
            binVec.allocateNew(n);

            boolVec.set(0, 1); boolVec.set(1, 0);
            int8Vec.set(0, 42); int8Vec.set(1, -1);
            int16Vec.set(0, 1234); int16Vec.set(1, -5678);
            int32Vec.set(0, 100000); int32Vec.set(1, -200000);
            int64Vec.set(0, 9999999999L); int64Vec.set(1, -9999999999L);
            f32Vec.set(0, 3.14f); f32Vec.set(1, -2.71f);
            f64Vec.set(0, 2.718281828); f64Vec.set(1, -3.141592653);
            utf8Vec.setSafe(0, "hello".getBytes()); utf8Vec.setSafe(1, "world".getBytes());
            binVec.setSafe(0, new byte[]{1, 2, 3}); binVec.setSafe(1, new byte[]{(byte) 0xff, 0});

            root.setRowCount(n);
            data = writeToBytes(arrowSchema, writer -> writer.write(root));
        }

        try (MosaicReader reader = readerFromBytes(data)) {
            try (VectorSchemaRoot batch = reader.readRowGroup(0, allocator)) {
                assertEquals(2, batch.getRowCount());
                assertEquals(1, ((BitVector) batch.getVector("f_bool")).get(0));
                assertEquals(0, ((BitVector) batch.getVector("f_bool")).get(1));
                assertEquals(42, ((TinyIntVector) batch.getVector("f_int8")).get(0));
                assertEquals(-1, ((TinyIntVector) batch.getVector("f_int8")).get(1));
                assertEquals(1234, ((SmallIntVector) batch.getVector("f_int16")).get(0));
                assertEquals(-5678, ((SmallIntVector) batch.getVector("f_int16")).get(1));
                assertEquals(100000, ((IntVector) batch.getVector("f_int32")).get(0));
                assertEquals(-200000, ((IntVector) batch.getVector("f_int32")).get(1));
                assertEquals(9999999999L, ((BigIntVector) batch.getVector("f_int64")).get(0));
                assertEquals(-9999999999L, ((BigIntVector) batch.getVector("f_int64")).get(1));
                assertEquals(3.14f, ((Float4Vector) batch.getVector("f_float32")).get(0), 1e-5f);
                assertEquals(-2.71f, ((Float4Vector) batch.getVector("f_float32")).get(1), 1e-5f);
                assertEquals(2.718281828, ((Float8Vector) batch.getVector("f_float64")).get(0), 1e-9);
                assertEquals(-3.141592653, ((Float8Vector) batch.getVector("f_float64")).get(1), 1e-9);
                assertEquals("hello", new String(((VarCharVector) batch.getVector("f_utf8")).get(0)));
                assertEquals("world", new String(((VarCharVector) batch.getVector("f_utf8")).get(1)));
                assertArrayEquals(new byte[]{1, 2, 3}, ((VarBinaryVector) batch.getVector("f_binary")).get(0));
                assertArrayEquals(new byte[]{(byte) 0xff, 0}, ((VarBinaryVector) batch.getVector("f_binary")).get(1));
            }
        }
    }

    @Test
    public void testCompressionNone() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("x", new ArrowType.Int(32, true)),
                Field.nullable("y", ArrowType.Utf8.INSTANCE)
        ));

        byte[] data;
        try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
            IntVector xVec = (IntVector) root.getVector("x");
            VarCharVector yVec = (VarCharVector) root.getVector("y");
            int n = 20;
            xVec.allocateNew(n);
            yVec.allocateNew(n);
            for (int i = 0; i < n; i++) {
                xVec.set(i, i);
                yVec.setSafe(i, ("v_" + i).getBytes());
            }
            root.setRowCount(n);
            data = writeToBytes(arrowSchema, new WriterOptions().compression(0), writer -> writer.write(root));
        }

        try (MosaicReader reader = readerFromBytes(data)) {
            try (VectorSchemaRoot batch = reader.readRowGroup(0, allocator)) {
                assertEquals(20, batch.getRowCount());
                for (int i = 0; i < 20; i++) {
                    assertEquals(i, ((IntVector) batch.getVector("x")).get(i));
                }
            }
        }
    }

    @Test
    public void testMultipleRowGroups() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("id", new ArrowType.Int(32, true)),
                Field.nullable("data", new ArrowType.Int(64, true))
        ));

        WriterOptions opts = new WriterOptions().compression(0).numBuckets(1).rowGroupMaxSize(200);

        byte[] data;
        int totalRows = 500;
        int batchSize = 10;
        data = writeToBytes(arrowSchema, opts, writer -> {
            for (int start = 0; start < totalRows; start += batchSize) {
                try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
                    IntVector idVec = (IntVector) root.getVector("id");
                    BigIntVector dataVec = (BigIntVector) root.getVector("data");
                    idVec.allocateNew(batchSize);
                    dataVec.allocateNew(batchSize);
                    for (int i = 0; i < batchSize; i++) {
                        idVec.set(i, start + i);
                        dataVec.set(i, (long) (start + i) * 3);
                    }
                    root.setRowCount(batchSize);
                    writer.write(root);
                }
            }
        });

        try (MosaicReader reader = readerFromBytes(data)) {
            assertTrue(reader.numRowGroups() > 1);
            int offset = 0;
            for (int rg = 0; rg < reader.numRowGroups(); rg++) {
                try (VectorSchemaRoot batch = reader.readRowGroup(rg, allocator)) {
                    IntVector ids = (IntVector) batch.getVector("id");
                    BigIntVector datas = (BigIntVector) batch.getVector("data");
                    for (int i = 0; i < batch.getRowCount(); i++) {
                        assertEquals(offset + i, ids.get(i));
                        assertEquals((long) (offset + i) * 3, datas.get(i));
                    }
                    offset += batch.getRowCount();
                }
            }
            assertEquals(500, offset);
        }
    }

    @Test
    public void testMultipleWrites() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("x", new ArrowType.Int(32, true))
        ));

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (MosaicWriter writer = new MosaicWriter(baos, arrowSchema, allocator)) {
            for (int start = 0; start < 30; start += 10) {
                try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
                    IntVector xVec = (IntVector) root.getVector("x");
                    xVec.allocateNew(10);
                    for (int i = 0; i < 10; i++) {
                        xVec.set(i, start + i);
                    }
                    root.setRowCount(10);
                    writer.write(root);
                }
            }
        }
        byte[] data = baos.toByteArray();

        try (MosaicReader reader = readerFromBytes(data)) {
            int totalRows = 0;
            for (int rg = 0; rg < reader.numRowGroups(); rg++) {
                try (VectorSchemaRoot batch = reader.readRowGroup(rg, allocator)) {
                    totalRows += batch.getRowCount();
                }
            }
            assertEquals(30, totalRows);
        }
    }

    @Test
    public void testWriteAfterCloseFailsBeforeExport() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("x", new ArrowType.Int(32, true))
        ));

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        MosaicWriter writer = new MosaicWriter(baos, arrowSchema, allocator);
        writer.close();

        try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
            assertThrows(IllegalStateException.class, () -> writer.write(root));
        }
    }

    @Test
    public void testSingleRow() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("v", new ArrowType.Int(32, true))
        ));

        byte[] data;
        try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
            IntVector vVec = (IntVector) root.getVector("v");
            vVec.allocateNew(1);
            vVec.set(0, 42);
            root.setRowCount(1);
            data = writeToBytes(arrowSchema, writer -> writer.write(root));
        }

        try (MosaicReader reader = readerFromBytes(data)) {
            try (VectorSchemaRoot batch = reader.readRowGroup(0, allocator)) {
                assertEquals(1, batch.getRowCount());
                assertEquals(42, ((IntVector) batch.getVector("v")).get(0));
            }
        }
    }

    @Test
    public void testZeroRows() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("v", new ArrowType.Int(32, true))
        ));

        byte[] data;
        try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
            root.getVector("v").allocateNew();
            root.setRowCount(0);
            data = writeToBytes(arrowSchema, writer -> writer.write(root));
        }

        try (MosaicReader reader = readerFromBytes(data)) {
            assertEquals(0, reader.numRowGroups());
        }
    }

    @Test
    public void testStatsWithNulls() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("a", new ArrowType.Int(32, true)),
                Field.nullable("b", new ArrowType.Int(64, true))
        ));

        WriterOptions opts = new WriterOptions().statsColumns(0, 1).numBuckets(1);

        byte[] data;
        try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
            IntVector aVec = (IntVector) root.getVector("a");
            BigIntVector bVec = (BigIntVector) root.getVector("b");
            aVec.allocateNew(4);
            bVec.allocateNew(4);

            aVec.set(0, 10);
            aVec.setNull(1);
            aVec.set(2, 5);
            aVec.set(3, 20);

            bVec.setNull(0);
            bVec.setNull(1);
            bVec.set(2, 100);
            bVec.set(3, 50);

            root.setRowCount(4);
            data = writeToBytes(arrowSchema, opts, writer -> writer.write(root));
        }

        try (MosaicReader reader = readerFromBytes(data)) {
            List<ColumnStatistics> stats = reader.getRowGroupStatistics(0);
            assertEquals(2, stats.size());

            ColumnStatistics aStat = stats.stream().filter(s -> s.getColumnIndex() == 0).findFirst().get();
            assertEquals(1, aStat.getNullCount());
            assertTrue(aStat.hasMinMax());
            int minA = ByteBuffer.wrap(aStat.getMin()).order(ByteOrder.BIG_ENDIAN).getInt();
            int maxA = ByteBuffer.wrap(aStat.getMax()).order(ByteOrder.BIG_ENDIAN).getInt();
            assertEquals(5, minA);
            assertEquals(20, maxA);

            ColumnStatistics bStat = stats.stream().filter(s -> s.getColumnIndex() == 1).findFirst().get();
            assertEquals(2, bStat.getNullCount());
            assertTrue(bStat.hasMinMax());
            long minB = ByteBuffer.wrap(bStat.getMin()).order(ByteOrder.BIG_ENDIAN).getLong();
            long maxB = ByteBuffer.wrap(bStat.getMax()).order(ByteOrder.BIG_ENDIAN).getLong();
            assertEquals(50, minB);
            assertEquals(100, maxB);
        }
    }

    @Test
    public void testStatsAllNull() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("x", new ArrowType.Int(32, true))
        ));

        WriterOptions opts = new WriterOptions().statsColumns(0).numBuckets(1);

        byte[] data;
        try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
            IntVector xVec = (IntVector) root.getVector("x");
            xVec.allocateNew(3);
            xVec.setNull(0);
            xVec.setNull(1);
            xVec.setNull(2);
            root.setRowCount(3);
            data = writeToBytes(arrowSchema, opts, writer -> writer.write(root));
        }

        try (MosaicReader reader = readerFromBytes(data)) {
            List<ColumnStatistics> stats = reader.getRowGroupStatistics(0);
            assertEquals(1, stats.size());
            assertEquals(3, stats.get(0).getNullCount());
            assertFalse(stats.get(0).hasMinMax());
        }
    }

    @Test
    public void testEstimatedFileSize() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("x", new ArrowType.Int(32, true)),
                Field.nullable("y", ArrowType.Utf8.INSTANCE)
        ));

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (MosaicWriter writer = new MosaicWriter(baos, arrowSchema, allocator)) {
            try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
                IntVector xVec = (IntVector) root.getVector("x");
                VarCharVector yVec = (VarCharVector) root.getVector("y");
                int n = 100;
                xVec.allocateNew(n);
                yVec.allocateNew(n);
                for (int i = 0; i < n; i++) {
                    xVec.set(i, i);
                    yVec.setSafe(i, ("value_" + i).getBytes());
                }
                root.setRowCount(n);
                writer.write(root);
            }
            assertTrue(writer.estimatedFileSize() > 0);
        }
    }

    @Test
    public void testSchemaRoundtrip() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.notNullable("id", new ArrowType.Int(32, true)),
                Field.nullable("name", ArrowType.Utf8.INSTANCE),
                Field.nullable("score", new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE))
        ));

        byte[] data;
        try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
            IntVector ids = (IntVector) root.getVector("id");
            VarCharVector names = (VarCharVector) root.getVector("name");
            Float8Vector scores = (Float8Vector) root.getVector("score");
            ids.allocateNew(1); names.allocateNew(1); scores.allocateNew(1);
            ids.set(0, 1); names.setSafe(0, "x".getBytes()); scores.set(0, 1.0);
            root.setRowCount(1);
            data = writeToBytes(arrowSchema, writer -> writer.write(root));
        }

        try (MosaicReader reader = readerFromBytes(data)) {
            Schema readSchema = reader.getSchema();
            assertEquals(3, readSchema.getFields().size());
            assertEquals("id", readSchema.getFields().get(0).getName());
            assertEquals("name", readSchema.getFields().get(1).getName());
            assertEquals("score", readSchema.getFields().get(2).getName());
            assertFalse(readSchema.getFields().get(0).isNullable());
            assertTrue(readSchema.getFields().get(1).isNullable());
        }
    }
}
