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
import java.util.Arrays;
import java.util.Collections;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

public class BloomFilterTest {

    private BufferAllocator allocator;

    @Before
    public void setUp() {
        allocator = new RootAllocator();
    }

    @After
    public void tearDown() {
        allocator.close();
    }

    private MosaicReader readerFromBytes(byte[] data) {
        InputFile inputFile = (position, buffer, offset, length) ->
                System.arraycopy(data, (int) position, buffer, offset, length);
        return MosaicReader.open(inputFile, data.length, allocator);
    }

    @Test
    public void bigIntColumnHitsAndMisses() {
        Schema schema = new Schema(Collections.singletonList(
                Field.notNullable("id", new ArrowType.Int(64, true))));
        WriterOptions opts = new WriterOptions()
                .numBuckets(2)
                .bloomFilterColumns(Collections.singletonList(
                        new BloomFilterConfig("id", 2000, 0.01)));

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (MosaicWriter writer = new MosaicWriter(baos, schema, opts, allocator);
             VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
            BigIntVector id = (BigIntVector) root.getVector("id");
            int n = 2000;
            id.allocateNew(n);
            for (int i = 0; i < n; i++) {
                id.set(i, i);
            }
            root.setRowCount(n);
            writer.write(root);
        }
        byte[] bytes = baos.toByteArray();

        try (MosaicReader reader = readerFromBytes(bytes)) {
            for (long i = 0; i < 2000; i++) {
                assertTrue("present value " + i + " missed", reader.bloomMightContain(0, "id", i));
            }
            int falsePositives = 0;
            int probe = 5000;
            for (long i = 1_000_000; i < 1_000_000 + probe; i++) {
                if (reader.bloomMightContain(0, "id", i)) {
                    falsePositives++;
                }
            }
            double rate = (double) falsePositives / probe;
            assertTrue("fpp " + rate + " above 5%", rate < 0.05);
        }
    }

    @Test
    public void stringColumnRejectsAbsentValue() {
        Schema schema = new Schema(Collections.singletonList(
                Field.notNullable("name", ArrowType.Utf8.INSTANCE)));
        WriterOptions opts = new WriterOptions()
                .numBuckets(1)
                .bloomFilterColumns(Collections.singletonList(
                        new BloomFilterConfig("name", 1024, 0.001)));

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        String[] names = new String[] {"alice", "bob", "carol", "dave", "eve"};
        try (MosaicWriter writer = new MosaicWriter(baos, schema, opts, allocator);
             VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
            VarCharVector vec = (VarCharVector) root.getVector("name");
            vec.allocateNew(names.length);
            for (int i = 0; i < names.length; i++) {
                vec.setSafe(i, names[i].getBytes(java.nio.charset.StandardCharsets.UTF_8));
            }
            root.setRowCount(names.length);
            writer.write(root);
        }
        byte[] bytes = baos.toByteArray();

        try (MosaicReader reader = readerFromBytes(bytes)) {
            for (String n : names) {
                assertTrue("present name " + n + " missed", reader.bloomMightContain(0, "name", n));
            }
            assertFalse("absent name unexpectedly present", reader.bloomMightContain(0, "name", "zachary"));
        }
    }

    @Test
    public void noBloomReturnsTrueConservatively() {
        Schema schema = new Schema(Collections.singletonList(
                Field.notNullable("id", new ArrowType.Int(64, true))));
        WriterOptions opts = new WriterOptions().numBuckets(1);

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (MosaicWriter writer = new MosaicWriter(baos, schema, opts, allocator);
             VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
            BigIntVector id = (BigIntVector) root.getVector("id");
            id.allocateNew(3);
            id.set(0, 1L);
            id.set(1, 2L);
            id.set(2, 3L);
            root.setRowCount(3);
            writer.write(root);
        }
        byte[] bytes = baos.toByteArray();

        try (MosaicReader reader = readerFromBytes(bytes)) {
            assertTrue(reader.bloomMightContain(0, "id", 99999L));
        }
    }
}
