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
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.math.BigDecimal;
import java.util.Arrays;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.DateDayVector;
import org.apache.arrow.vector.DecimalVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.SmallIntVector;
import org.apache.arrow.vector.TinyIntVector;
import org.apache.arrow.vector.VarBinaryVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.DateUnit;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Cross-language interoperability tests.
 *
 * These tests read .mosaic files written by the Rust interop_write_test,
 * verifying that the Java binding can correctly read files produced by Rust.
 * One test also writes a file from Java for Rust to read back.
 */
public class MosaicInteropTest {

    private static final String INTEROP_DIR = "/tmp/mosaic_interop";
    private BufferAllocator allocator;

    @Before
    public void setUp() {
        allocator = new RootAllocator();
    }

    @After
    public void tearDown() {
        allocator.close();
    }

    /** InputFile backed by a RandomAccessFile for reading .mosaic files from disk. */
    private static class FileInputFile implements InputFile {
        private final RandomAccessFile raf;

        FileInputFile(String path) throws IOException {
            this.raf = new RandomAccessFile(path, "r");
        }

        @Override
        public void readFully(long position, byte[] buffer, int offset, int length) throws IOException {
            synchronized (raf) {
                raf.seek(position);
                raf.readFully(buffer, offset, length);
            }
        }

        long length() throws IOException {
            return raf.length();
        }

        void close() throws IOException {
            raf.close();
        }
    }

    private MosaicReader openFile(String filename) throws IOException {
        String path = INTEROP_DIR + "/" + filename;
        FileInputFile inputFile = new FileInputFile(path);
        long fileLength = inputFile.length();
        return MosaicReader.open(inputFile, fileLength, allocator);
    }

    // ======================== 1. Read int_data.mosaic ========================

    @Test
    public void testReadRustIntData() throws IOException {
        try (MosaicReader reader = openFile("int_data.mosaic")) {
            assertEquals(2, reader.getSchema().getFields().size());

            int totalRows = 0;
            for (int rg = 0; rg < reader.numRowGroups(); rg++) {
                try (VectorSchemaRoot batch = reader.readRowGroup(rg, allocator)) {
                    BigIntVector ids = (BigIntVector) batch.getVector("id");
                    IntVector values = (IntVector) batch.getVector("value");

                    for (int i = 0; i < batch.getRowCount(); i++) {
                        int rowIdx = totalRows + i;
                        assertEquals("id mismatch at row " + rowIdx, (long) rowIdx, ids.get(i));
                        assertEquals("value mismatch at row " + rowIdx, rowIdx * 10, values.get(i));
                    }
                    totalRows += batch.getRowCount();
                }
            }
            assertEquals(10000, totalRows);
        }
        System.out.println("testReadRustIntData: PASSED (10000 rows)");
    }

    // ======================== 2. Read string_data.mosaic ========================

    @Test
    public void testReadRustStringData() throws IOException {
        try (MosaicReader reader = openFile("string_data.mosaic")) {
            assertEquals(3, reader.getSchema().getFields().size());

            int totalRows = 0;
            int nameNulls = 0;
            int dataNulls = 0;
            for (int rg = 0; rg < reader.numRowGroups(); rg++) {
                try (VectorSchemaRoot batch = reader.readRowGroup(rg, allocator)) {
                    BigIntVector ids = (BigIntVector) batch.getVector("id");
                    VarCharVector names = (VarCharVector) batch.getVector("name");
                    VarBinaryVector data = (VarBinaryVector) batch.getVector("data");

                    for (int i = 0; i < batch.getRowCount(); i++) {
                        int rowIdx = totalRows + i;
                        assertEquals((long) rowIdx, ids.get(i));

                        if (rowIdx % 7 == 0) {
                            assertTrue("name should be null at row " + rowIdx, names.isNull(i));
                            nameNulls++;
                        } else {
                            assertFalse("name should not be null at row " + rowIdx, names.isNull(i));
                            assertEquals("name_" + rowIdx, new String(names.get(i)));
                        }

                        if (rowIdx % 5 == 0) {
                            assertTrue("data should be null at row " + rowIdx, data.isNull(i));
                            dataNulls++;
                        } else {
                            assertFalse("data should not be null at row " + rowIdx, data.isNull(i));
                            assertArrayEquals(("bin_" + rowIdx).getBytes(), data.get(i));
                        }
                    }
                    totalRows += batch.getRowCount();
                }
            }
            assertEquals(10000, totalRows);
            // Verify null counts: every 7th row is null for name
            assertTrue("Expected name nulls > 0, got " + nameNulls, nameNulls > 0);
            assertTrue("Expected data nulls > 0, got " + dataNulls, dataNulls > 0);
        }
        System.out.println("testReadRustStringData: PASSED");
    }

    // ======================== 3. Read all_types.mosaic ========================

    @Test
    public void testReadRustAllTypes() throws IOException {
        try (MosaicReader reader = openFile("all_types.mosaic")) {
            assertEquals(11, reader.getSchema().getFields().size());

            int totalRows = 0;
            for (int rg = 0; rg < reader.numRowGroups(); rg++) {
                try (VectorSchemaRoot batch = reader.readRowGroup(rg, allocator)) {
                    BitVector bools = (BitVector) batch.getVector("f_bool");
                    TinyIntVector i8s = (TinyIntVector) batch.getVector("f_int8");
                    SmallIntVector i16s = (SmallIntVector) batch.getVector("f_int16");
                    IntVector i32s = (IntVector) batch.getVector("f_int32");
                    BigIntVector i64s = (BigIntVector) batch.getVector("f_int64");
                    Float4Vector f32s = (Float4Vector) batch.getVector("f_float32");
                    Float8Vector f64s = (Float8Vector) batch.getVector("f_float64");
                    DateDayVector dates = (DateDayVector) batch.getVector("f_date32");
                    VarCharVector strs = (VarCharVector) batch.getVector("f_utf8");
                    VarBinaryVector bins = (VarBinaryVector) batch.getVector("f_binary");
                    DecimalVector decs = (DecimalVector) batch.getVector("f_decimal");

                    for (int i = 0; i < batch.getRowCount(); i++) {
                        int rowIdx = totalRows + i;

                        // Boolean: null every 13th
                        if (rowIdx % 13 == 0) {
                            assertTrue(bools.isNull(i));
                        } else {
                            assertFalse(bools.isNull(i));
                            assertEquals(rowIdx % 2 == 0 ? 1 : 0, bools.get(i));
                        }

                        // Int8: null every 11th
                        if (rowIdx % 11 == 0) {
                            assertTrue(i8s.isNull(i));
                        } else {
                            assertFalse(i8s.isNull(i));
                            assertEquals((byte) (rowIdx % 256), i8s.get(i));
                        }

                        // Int16: null every 17th
                        if (rowIdx % 17 == 0) {
                            assertTrue(i16s.isNull(i));
                        } else {
                            assertFalse(i16s.isNull(i));
                            assertEquals((short) (rowIdx % 30000), i16s.get(i));
                        }

                        // Int32: null every 19th
                        if (rowIdx % 19 == 0) {
                            assertTrue(i32s.isNull(i));
                        } else {
                            assertFalse(i32s.isNull(i));
                            assertEquals(rowIdx * 100, i32s.get(i));
                        }

                        // Int64: null every 23rd
                        if (rowIdx % 23 == 0) {
                            assertTrue(i64s.isNull(i));
                        } else {
                            assertFalse(i64s.isNull(i));
                            assertEquals((long) rowIdx * 1000, i64s.get(i));
                        }

                        // Float32: null every 29th
                        if (rowIdx % 29 == 0) {
                            assertTrue(f32s.isNull(i));
                        } else {
                            assertFalse(f32s.isNull(i));
                            assertEquals(rowIdx * 0.1f, f32s.get(i), 1e-5f);
                        }

                        // Float64: null every 31st
                        if (rowIdx % 31 == 0) {
                            assertTrue(f64s.isNull(i));
                        } else {
                            assertFalse(f64s.isNull(i));
                            assertEquals(rowIdx * 0.001, f64s.get(i), 1e-9);
                        }

                        // Date32: null every 37th
                        if (rowIdx % 37 == 0) {
                            assertTrue(dates.isNull(i));
                        } else {
                            assertFalse(dates.isNull(i));
                            assertEquals(18000 + (rowIdx % 3650), dates.get(i));
                        }

                        // Utf8: null every 41st
                        if (rowIdx % 41 == 0) {
                            assertTrue(strs.isNull(i));
                        } else {
                            assertFalse(strs.isNull(i));
                            assertEquals("str_" + rowIdx, new String(strs.get(i)));
                        }

                        // Binary: null every 43rd
                        if (rowIdx % 43 == 0) {
                            assertTrue(bins.isNull(i));
                        } else {
                            assertFalse(bins.isNull(i));
                            byte[] expected = new byte[4];
                            java.util.Arrays.fill(expected, (byte) (rowIdx % 256));
                            assertArrayEquals(expected, bins.get(i));
                        }

                        // Decimal128(10,2): null every 47th
                        if (rowIdx % 47 == 0) {
                            assertTrue(decs.isNull(i));
                        } else {
                            assertFalse(decs.isNull(i));
                            BigDecimal expected = new BigDecimal(rowIdx).setScale(0)
                                    .multiply(new BigDecimal("1.00"));
                            assertEquals(expected, decs.getObject(i));
                        }
                    }
                    totalRows += batch.getRowCount();
                }
            }
            assertEquals(5000, totalRows);
        }
        System.out.println("testReadRustAllTypes: PASSED (5000 rows, 11 columns)");
    }

    // ======================== 4. Read constant_data.mosaic ========================

    @Test
    public void testReadRustConstantData() throws IOException {
        try (MosaicReader reader = openFile("constant_data.mosaic")) {
            int totalRows = 0;
            for (int rg = 0; rg < reader.numRowGroups(); rg++) {
                try (VectorSchemaRoot batch = reader.readRowGroup(rg, allocator)) {
                    BigIntVector ints = (BigIntVector) batch.getVector("c_int");
                    VarCharVector strs = (VarCharVector) batch.getVector("c_str");
                    Float8Vector floats = (Float8Vector) batch.getVector("c_float");

                    for (int i = 0; i < batch.getRowCount(); i++) {
                        assertEquals(42L, ints.get(i));
                        assertEquals("constant_value", new String(strs.get(i)));
                        assertEquals(3.14, floats.get(i), 1e-9);
                    }
                    totalRows += batch.getRowCount();
                }
            }
            assertEquals(10000, totalRows);
        }
        System.out.println("testReadRustConstantData: PASSED (10000 rows, all constant)");
    }

    // ======================== 5. Read null_heavy.mosaic ========================

    @Test
    public void testReadRustNullHeavy() throws IOException {
        try (MosaicReader reader = openFile("null_heavy.mosaic")) {
            int totalRows = 0;
            int intNulls = 0;
            int strNulls = 0;
            int floatNulls = 0;
            for (int rg = 0; rg < reader.numRowGroups(); rg++) {
                try (VectorSchemaRoot batch = reader.readRowGroup(rg, allocator)) {
                    BigIntVector ints = (BigIntVector) batch.getVector("n_int64");
                    VarCharVector strs = (VarCharVector) batch.getVector("n_utf8");
                    Float8Vector floats = (Float8Vector) batch.getVector("n_float64");

                    for (int i = 0; i < batch.getRowCount(); i++) {
                        int rowIdx = totalRows + i;

                        // int: non-null when rowIdx % 5 == 0
                        if (rowIdx % 5 != 0) {
                            assertTrue(ints.isNull(i));
                            intNulls++;
                        } else {
                            assertFalse(ints.isNull(i));
                            assertEquals((long) rowIdx, ints.get(i));
                        }

                        // str: non-null when rowIdx % 5 == 1
                        if (rowIdx % 5 != 1) {
                            assertTrue(strs.isNull(i));
                            strNulls++;
                        } else {
                            assertFalse(strs.isNull(i));
                            assertEquals("val_" + rowIdx, new String(strs.get(i)));
                        }

                        // float: non-null when rowIdx % 5 == 2
                        if (rowIdx % 5 != 2) {
                            assertTrue(floats.isNull(i));
                            floatNulls++;
                        } else {
                            assertFalse(floats.isNull(i));
                            assertEquals(rowIdx * 0.5, floats.get(i), 1e-9);
                        }
                    }
                    totalRows += batch.getRowCount();
                }
            }
            assertEquals(10000, totalRows);
            // 80% null for each column
            assertEquals(8000, intNulls);
            assertEquals(8000, strNulls);
            assertEquals(8000, floatNulls);
        }
        System.out.println("testReadRustNullHeavy: PASSED (10000 rows, 80% nulls)");
    }

    // ======================== 6. Read compressed_none.mosaic ========================

    @Test
    public void testReadRustNoCompression() throws IOException {
        try (MosaicReader reader = openFile("compressed_none.mosaic")) {
            int totalRows = 0;
            for (int rg = 0; rg < reader.numRowGroups(); rg++) {
                try (VectorSchemaRoot batch = reader.readRowGroup(rg, allocator)) {
                    BigIntVector ids = (BigIntVector) batch.getVector("id");
                    IntVector values = (IntVector) batch.getVector("value");

                    for (int i = 0; i < batch.getRowCount(); i++) {
                        int rowIdx = totalRows + i;
                        assertEquals((long) rowIdx, ids.get(i));
                        assertEquals(rowIdx * 10, values.get(i));
                    }
                    totalRows += batch.getRowCount();
                }
            }
            assertEquals(10000, totalRows);
        }
        System.out.println("testReadRustNoCompression: PASSED");
    }

    // ======================== 7. Read multi_rg.mosaic ========================

    @Test
    public void testReadRustMultiRowGroup() throws IOException {
        try (MosaicReader reader = openFile("multi_rg.mosaic")) {
            assertTrue("Expected multiple row groups", reader.numRowGroups() > 1);

            int totalRows = 0;
            for (int rg = 0; rg < reader.numRowGroups(); rg++) {
                try (VectorSchemaRoot batch = reader.readRowGroup(rg, allocator)) {
                    BigIntVector ids = (BigIntVector) batch.getVector("id");
                    IntVector values = (IntVector) batch.getVector("value");

                    for (int i = 0; i < batch.getRowCount(); i++) {
                        int rowIdx = totalRows + i;
                        assertEquals((long) rowIdx, ids.get(i));
                        assertEquals(rowIdx * 10, values.get(i));
                    }
                    totalRows += batch.getRowCount();
                }
            }
            assertEquals(10000, totalRows);
        }
        System.out.println("testReadRustMultiRowGroup: PASSED (multiple row groups)");
    }

    // ======================== 8. Write from Java, read from Rust ========================

    @Test
    public void testWriteFileReadFromRust() throws IOException {
        // Write a mosaic file from Java to be read by Rust
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.notNullable("id", new ArrowType.Int(64, true)),
                Field.nullable("name", ArrowType.Utf8.INSTANCE),
                Field.notNullable("score", new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE))
        ));

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        int numRows = 5000;
        try (MosaicWriter writer = new MosaicWriter(baos, arrowSchema, new WriterOptions().numBuckets(1), allocator)) {
            try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
                BigIntVector ids = (BigIntVector) root.getVector("id");
                VarCharVector names = (VarCharVector) root.getVector("name");
                Float8Vector scores = (Float8Vector) root.getVector("score");

                ids.allocateNew(numRows);
                names.allocateNew(numRows);
                scores.allocateNew(numRows);

                for (int i = 0; i < numRows; i++) {
                    ids.set(i, (long) i);
                    if (i % 3 == 0) {
                        names.setNull(i);
                    } else {
                        names.setSafe(i, ("java_name_" + i).getBytes());
                    }
                    scores.set(i, i * 2.5);
                }
                root.setRowCount(numRows);
                writer.write(root);
            }
        }

        byte[] data = baos.toByteArray();
        assertTrue("File data should not be empty", data.length > 0);

        // Write to disk
        File dir = new File(INTEROP_DIR);
        dir.mkdirs();
        String path = INTEROP_DIR + "/java_written.mosaic";
        try (FileOutputStream fos = new FileOutputStream(path)) {
            fos.write(data);
        }

        // Immediately verify we can read it back in Java
        InputFile inputFile = (position, buffer, offset, length) -> {
            System.arraycopy(data, (int) position, buffer, offset, length);
        };
        try (MosaicReader reader = MosaicReader.open(inputFile, data.length, allocator)) {
            int totalRows = 0;
            for (int rg = 0; rg < reader.numRowGroups(); rg++) {
                try (VectorSchemaRoot batch = reader.readRowGroup(rg, allocator)) {
                    BigIntVector ids = (BigIntVector) batch.getVector("id");
                    VarCharVector names = (VarCharVector) batch.getVector("name");
                    Float8Vector scores = (Float8Vector) batch.getVector("score");

                    for (int i = 0; i < batch.getRowCount(); i++) {
                        int rowIdx = totalRows + i;
                        assertEquals((long) rowIdx, ids.get(i));
                        if (rowIdx % 3 == 0) {
                            assertTrue(names.isNull(i));
                        } else {
                            assertEquals("java_name_" + rowIdx, new String(names.get(i)));
                        }
                        assertEquals(rowIdx * 2.5, scores.get(i), 1e-9);
                    }
                    totalRows += batch.getRowCount();
                }
            }
            assertEquals(numRows, totalRows);
        }
        System.out.println("testWriteFileReadFromRust: PASSED (" + numRows + " rows written to " + path + ")");
    }
}
