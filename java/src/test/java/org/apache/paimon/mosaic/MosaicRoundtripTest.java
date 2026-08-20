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
import java.io.IOException;
import java.io.OutputStream;
import java.lang.ref.WeakReference;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.arrow.c.jni.JniWrapper;
import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.OutOfMemoryException;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.BitVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float4Vector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.SmallIntVector;
import org.apache.arrow.vector.TimeStampNanoTZVector;
import org.apache.arrow.vector.TimeStampNanoVector;
import org.apache.arrow.vector.TinyIntVector;
import org.apache.arrow.vector.VarBinaryVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.complex.ListVector;
import org.apache.arrow.vector.complex.MapVector;
import org.apache.arrow.vector.complex.impl.UnionListWriter;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

public class MosaicRoundtripTest {

    private BufferAllocator allocator;

    private static final class InjectableListVector extends ListVector {

        private InjectableListVector(String name, BufferAllocator allocator) {
            super(name, allocator, FieldType.nullable(ArrowType.List.INSTANCE), null);
        }

        private void setDataVector(FieldVector vector) {
            replaceDataVector(vector);
        }
    }

    private static final class InjectedExportException extends RuntimeException {}

    private static final class FailOnceListVector extends ListVector {

        private boolean fail = true;

        private FailOnceListVector(String name, BufferAllocator allocator) {
            super(name, allocator, FieldType.nullable(ArrowType.List.INSTANCE), null);
            replaceDataVector(new IntVector(ListVector.DATA_VECTOR_NAME, allocator));
        }

        @Override
        public List<ArrowBuf> getFieldBuffers() {
            if (fail) {
                fail = false;
                throw new InjectedExportException();
            }
            return super.getFieldBuffers();
        }
    }

    private enum FailurePoint {
        WRITE,
        FLUSH
    }

    private static final class FailOnceOutputStream extends OutputStream {

        private final ByteArrayOutputStream delegate = new ByteArrayOutputStream();
        private final FailurePoint failurePoint;
        private final IllegalStateException failureCause;
        private final IOException failure;
        private boolean failed;
        private int writeCalls;
        private int flushCalls;

        private FailOnceOutputStream(FailurePoint failurePoint, String message) {
            this.failurePoint = failurePoint;
            this.failureCause = new IllegalStateException(message + "-cause");
            this.failure = new IOException(message, failureCause);
        }

        @Override
        public void write(int value) throws IOException {
            write(new byte[] {(byte) value}, 0, 1);
        }

        @Override
        public void write(byte[] bytes, int offset, int length) throws IOException {
            writeCalls++;
            if (!failed && failurePoint == FailurePoint.WRITE) {
                failed = true;
                throw failure;
            }
            delegate.write(bytes, offset, length);
        }

        @Override
        public void flush() throws IOException {
            flushCalls++;
            if (!failed && failurePoint == FailurePoint.FLUSH) {
                failed = true;
                throw failure;
            }
        }

        private int size() {
            return delegate.size();
        }
    }

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

    private static Schema wideIntSchema(int width) {
        List<Field> fields = new ArrayList<>(width);
        for (int i = 0; i < width; i++) {
            fields.add(Field.nullable("c" + i, new ArrowType.Int(32, true)));
        }
        return new Schema(fields);
    }

    private static void awaitGarbageCollection(WeakReference<?> reference) throws InterruptedException {
        for (int i = 0; i < 20 && reference.get() != null; i++) {
            System.gc();
            System.runFinalization();
            Thread.sleep(50L);
        }
        assertNull("expected native callback object to be released", reference.get());
    }

    private static void awaitGarbageCollection(List<WeakReference<?>> references)
            throws InterruptedException {
        for (WeakReference<?> reference : references) {
            awaitGarbageCollection(reference);
        }
    }

    private WeakReference<InputFile> openReaderWithClosedAllocator(byte[] data) {
        BufferAllocator failingAllocator = new RootAllocator();
        failingAllocator.close();

        InputFile inputFile = new InputFile() {
            @Override
            public void readFully(long position, byte[] buffer, int offset, int length) {
                System.arraycopy(data, (int) position, buffer, offset, length);
            }
        };
        WeakReference<InputFile> reference = new WeakReference<>(inputFile);

        assertThrows(RuntimeException.class, () -> MosaicReader.open(inputFile, data.length, failingAllocator));
        return reference;
    }

    private List<WeakReference<?>> openReaderWithFailingInput() {
        IOException expected = new IOException("intentional native input failure");
        InputFile inputFile =
                new InputFile() {
                    @Override
                    public void readFully(
                            long position, byte[] buffer, int offset, int length)
                            throws IOException {
                        throw expected;
                    }
                };
        WeakReference<InputFile> inputReference = new WeakReference<>(inputFile);
        WeakReference<IOException> exceptionReference = new WeakReference<>(expected);

        IOException error =
                assertThrows(
                        IOException.class,
                        () -> MosaicReader.open(inputFile, 64L, allocator));
        assertSame(expected, error);
        return Arrays.asList(inputReference, exceptionReference);
    }

    private List<WeakReference<?>> readRowGroupWithFailingInput(byte[] data) {
        IOException expected = new IOException("intentional native background input failure");
        long callingThreadId = Thread.currentThread().getId();
        AtomicBoolean failReads = new AtomicBoolean();
        AtomicInteger reads = new AtomicInteger();
        AtomicLong failingThreadId = new AtomicLong(-1L);
        InputFile inputFile =
                new InputFile() {
                    @Override
                    public void readFully(
                            long position, byte[] buffer, int offset, int length)
                            throws IOException {
                        reads.incrementAndGet();
                        if (failReads.get()) {
                            failingThreadId.compareAndSet(
                                    -1L, Thread.currentThread().getId());
                            throw expected;
                        }
                        System.arraycopy(data, (int) position, buffer, offset, length);
                    }
                };
        WeakReference<InputFile> inputReference = new WeakReference<>(inputFile);
        WeakReference<IOException> exceptionReference = new WeakReference<>(expected);

        long handle = NativeLib.nativeReaderOpen(inputFile, data.length);
        assertNotEquals(0L, handle);
        int readsAfterOpen = reads.get();
        try {
            failReads.set(true);
            IOException actual =
                    assertThrows(
                            IOException.class,
                            () -> NativeLib.nativeReaderOpenRowGroup(handle, 0));
            assertSame(expected, actual);
            assertTrue("expected a row-group read", reads.get() > readsAfterOpen);
            assertNotEquals(callingThreadId, failingThreadId.get());
            assertEquals(1, NativeLib.nativeReaderNumRowGroups(handle));
        } finally {
            NativeLib.nativeReaderFree(handle);
        }
        return Arrays.asList(inputReference, exceptionReference);
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
    public void testWriteFromIndependentRootAllocator() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.notNullable("id", new ArrowType.Int(32, true)),
                Field.nullable("name", ArrowType.Utf8.INSTANCE)
        ));

        byte[] data;
        try (BufferAllocator inputAllocator = new RootAllocator();
             VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, inputAllocator)) {
            IntVector ids = (IntVector) root.getVector("id");
            VarCharVector names = (VarCharVector) root.getVector("name");

            ids.allocateNew(3);
            names.allocateNew(3);
            for (int i = 0; i < 3; i++) {
                ids.set(i, i + 1);
                names.setSafe(i, ("input_" + i).getBytes());
            }
            root.setRowCount(3);

            data = writeToBytes(arrowSchema, writer -> writer.write(root));
        }

        try (MosaicReader reader = readerFromBytes(data);
             VectorSchemaRoot batch = reader.readRowGroup(0, allocator)) {
            assertEquals(3, batch.getRowCount());
            IntVector ids = (IntVector) batch.getVector("id");
            VarCharVector names = (VarCharVector) batch.getVector("name");
            for (int i = 0; i < 3; i++) {
                assertEquals(i + 1, ids.get(i));
                assertEquals("input_" + i, new String(names.get(i)));
            }
        }
    }

    @Test
    public void testWriteFromLimitedChildAllocator() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.notNullable("id", new ArrowType.Int(32, true))
        ));

        byte[] data;
        try (BufferAllocator inputRoot = new RootAllocator();
             BufferAllocator limitedAllocator =
                     inputRoot.newChildAllocator("limited-input", 0, 512);
             VectorSchemaRoot root =
                     VectorSchemaRoot.create(arrowSchema, limitedAllocator)) {
            IntVector ids = (IntVector) root.getVector("id");
            ids.allocateNew(1);
            ids.set(0, 7);
            root.setRowCount(1);

            long inputBytes = inputRoot.getAllocatedMemory();
            long inputPeak = limitedAllocator.getPeakMemoryAllocation();
            int inputChildren = inputRoot.getChildAllocators().size();
            ByteArrayOutputStream output = new ByteArrayOutputStream();
            try (MosaicWriter writer =
                    new MosaicWriter(output, arrowSchema, allocator)) {
                writer.write(root);
                assertEquals(inputBytes, inputRoot.getAllocatedMemory());
                assertEquals(inputPeak, limitedAllocator.getPeakMemoryAllocation());
                assertEquals(inputChildren, inputRoot.getChildAllocators().size());
            }
            data = output.toByteArray();
        }

        try (MosaicReader reader = readerFromBytes(data);
             VectorSchemaRoot batch = reader.readRowGroup(0, allocator)) {
            assertEquals(1, batch.getRowCount());
            assertEquals(7, ((IntVector) batch.getVector("id")).get(0));
        }
    }

    @Test
    public void testCrossRootExportUsesWriterAllocatorAndReleasesMetadata() {
        Schema arrowSchema = wideIntSchema(5_000);

        byte[] data;
        try (RootAllocator inputRoot = new RootAllocator(16L * 1024 * 1024);
             BufferAllocator limitedAllocator =
                     inputRoot.newChildAllocator("limited-input", 0, 512);
             VectorSchemaRoot root =
                     VectorSchemaRoot.create(arrowSchema, limitedAllocator)) {
            root.setRowCount(0);
            long inputBytes = inputRoot.getAllocatedMemory();
            long inputPeak = inputRoot.getPeakMemoryAllocation();
            int inputChildren = inputRoot.getChildAllocators().size();
            long writerBytes = allocator.getAllocatedMemory();

            ByteArrayOutputStream output = new ByteArrayOutputStream();
            try (MosaicWriter writer =
                    new MosaicWriter(output, arrowSchema, allocator)) {
                long writerPeak = allocator.getPeakMemoryAllocation();
                writer.write(root);
                assertEquals(inputBytes, inputRoot.getAllocatedMemory());
                assertEquals(inputPeak, inputRoot.getPeakMemoryAllocation());
                assertEquals(inputChildren, inputRoot.getChildAllocators().size());
                assertEquals(writerBytes, allocator.getAllocatedMemory());
                assertTrue(
                        "expected Arrow C Data metadata on the writer allocator",
                        allocator.getPeakMemoryAllocation() > writerPeak);
            }
            data = output.toByteArray();
        }

        assertTrue(data.length > 32);
    }

    @Test
    public void testSameRootExportKeepsWriterAllocatorAccounting() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.notNullable("id", new ArrowType.Int(32, true))
        ));

        byte[] data;
        try (RootAllocator sharedRoot = new RootAllocator(16L * 1024 * 1024);
             BufferAllocator writerAllocator =
                     sharedRoot.newChildAllocator("writer", 0, 16L * 1024 * 1024);
             BufferAllocator inputAllocator =
                     sharedRoot.newChildAllocator("input", 0, 16L * 1024 * 1024);
             VectorSchemaRoot root =
                     VectorSchemaRoot.create(arrowSchema, inputAllocator)) {
            IntVector ids = (IntVector) root.getVector("id");
            ids.allocateNew(1);
            ids.set(0, 7);
            root.setRowCount(1);

            long inputBytes = inputAllocator.getAllocatedMemory();
            long inputPeak = inputAllocator.getPeakMemoryAllocation();

            ByteArrayOutputStream output = new ByteArrayOutputStream();
            try (MosaicWriter writer =
                    new MosaicWriter(output, arrowSchema, writerAllocator)) {
                long writerPeak = writerAllocator.getPeakMemoryAllocation();
                writer.write(root);
                assertTrue(
                        "expected Arrow C Data metadata on the writer allocator",
                        writerAllocator.getPeakMemoryAllocation() > writerPeak);
                assertEquals(inputBytes, inputAllocator.getAllocatedMemory());
                assertEquals(inputPeak, inputAllocator.getPeakMemoryAllocation());
            }
            data = output.toByteArray();
        }

        try (MosaicReader reader = readerFromBytes(data);
             VectorSchemaRoot batch = reader.readRowGroup(0, allocator)) {
            assertEquals(1, batch.getRowCount());
            assertEquals(7, ((IntVector) batch.getVector("id")).get(0));
        }
    }

    @Test
    public void testCrossRootWriterAllocatorOutOfMemoryCanRetryWithoutLeak() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("id", new ArrowType.Int(32, true))
        ));

        byte[] data;
        try (RootAllocator writerRoot = new RootAllocator(16L * 1024 * 1024);
             RootAllocator inputRoot = new RootAllocator(16L * 1024 * 1024)) {
            try (BufferAllocator writerAllocator =
                         writerRoot.newChildAllocator("writer", 0, 16L * 1024 * 1024);
                 BufferAllocator inputAllocator =
                         inputRoot.newChildAllocator("limited-input", 0, 16L * 1024 * 1024);
                 VectorSchemaRoot root =
                         VectorSchemaRoot.create(arrowSchema, inputAllocator)) {
                IntVector ids = (IntVector) root.getVector("id");
                int rowCount = 65_536;
                ids.allocateNew(rowCount);
                for (int i = 0; i < rowCount; i++) {
                    ids.set(i, i);
                }
                root.setRowCount(rowCount);

                long inputBytes = inputRoot.getAllocatedMemory();
                ByteArrayOutputStream output = new ByteArrayOutputStream();
                try (MosaicWriter writer =
                        new MosaicWriter(output, arrowSchema, writerAllocator)) {
                    long writerBytes = writerAllocator.getAllocatedMemory();
                    writerAllocator.setLimit(writerBytes + 512);
                    assertThrows(OutOfMemoryException.class, () -> writer.write(root));
                    assertEquals(inputBytes, inputRoot.getAllocatedMemory());
                    assertEquals(inputBytes, inputAllocator.getAllocatedMemory());
                    assertEquals(writerBytes, writerAllocator.getAllocatedMemory());
                    assertEquals(rowCount - 1, ids.get(rowCount - 1));

                    writerAllocator.setLimit(16L * 1024 * 1024);
                    writer.write(root);
                }
                data = output.toByteArray();
            }
            assertEquals(0, inputRoot.getAllocatedMemory());
            assertEquals(0, writerRoot.getAllocatedMemory());
        }

        try (MosaicReader reader = readerFromBytes(data);
             VectorSchemaRoot batch = reader.readRowGroup(0, allocator)) {
            assertEquals(65_536, batch.getRowCount());
            assertEquals(65_535, ((IntVector) batch.getVector("id")).get(65_535));
        }
    }

    @Test
    public void testCrossRootFailureAfterRootRegistrationCanRetryWithoutLeak() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.notNullable("id", new ArrowType.Int(32, true))
        ));
        OutOfMemoryError injected =
                new OutOfMemoryError("injected after root callback registration");
        boolean[] failOnce = {true};
        MosaicWriter.RootArrayExporter rootArrayExporter =
                (address, privateData) -> {
                    JniWrapper.get().exportArray(address, privateData);
                    if (failOnce[0]) {
                        failOnce[0] = false;
                        throw injected;
                    }
                };

        byte[] data;
        try (RootAllocator writerRoot = new RootAllocator(16L * 1024 * 1024);
             RootAllocator inputRoot = new RootAllocator(16L * 1024 * 1024);
             BufferAllocator writerAllocator =
                     writerRoot.newChildAllocator("writer", 0, 16L * 1024 * 1024);
             BufferAllocator inputAllocator =
                     inputRoot.newChildAllocator("input", 0, 16L * 1024 * 1024);
             VectorSchemaRoot root =
                     VectorSchemaRoot.create(arrowSchema, inputAllocator)) {
            IntVector ids = (IntVector) root.getVector("id");
            ids.allocateNew(1);
            ids.set(0, 7);
            root.setRowCount(1);

            long inputBytes = inputAllocator.getAllocatedMemory();
            List<ArrowBuf> fieldBuffers = ids.getFieldBuffers();
            int[] refCounts = new int[fieldBuffers.size()];
            for (int i = 0; i < fieldBuffers.size(); i++) {
                refCounts[i] = fieldBuffers.get(i).refCnt();
            }

            ByteArrayOutputStream output = new ByteArrayOutputStream();
            try (MosaicWriter writer =
                    new MosaicWriter(
                            output,
                            arrowSchema,
                            new WriterOptions(),
                            writerAllocator,
                            rootArrayExporter)) {
                long writerBytes = writerAllocator.getAllocatedMemory();
                assertSame(injected, assertThrows(OutOfMemoryError.class, () -> writer.write(root)));
                assertEquals(inputBytes, inputAllocator.getAllocatedMemory());
                assertEquals(writerBytes, writerAllocator.getAllocatedMemory());
                for (int i = 0; i < fieldBuffers.size(); i++) {
                    assertEquals(refCounts[i], fieldBuffers.get(i).refCnt());
                }

                writer.write(root);
                assertEquals(inputBytes, inputAllocator.getAllocatedMemory());
                assertEquals(writerBytes, writerAllocator.getAllocatedMemory());
                for (int i = 0; i < fieldBuffers.size(); i++) {
                    assertEquals(refCounts[i], fieldBuffers.get(i).refCnt());
                }
            }
            data = output.toByteArray();
        }

        try (MosaicReader reader = readerFromBytes(data);
             VectorSchemaRoot batch = reader.readRowGroup(0, allocator)) {
            assertEquals(1, batch.getRowCount());
            assertEquals(7, ((IntVector) batch.getVector("id")).get(0));
        }
    }

    @Test
    public void testCrossRootPreflightValidationFailureCanRetryWithoutLeak() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.notNullable("id", new ArrowType.Int(32, true))
        ));

        byte[] data;
        try (RootAllocator writerRoot = new RootAllocator(16L * 1024 * 1024);
             RootAllocator inputRoot = new RootAllocator(16L * 1024 * 1024);
             BufferAllocator writerAllocator =
                     writerRoot.newChildAllocator("writer", 0, 16L * 1024 * 1024);
             BufferAllocator inputAllocator =
                     inputRoot.newChildAllocator("input", 0, 16L * 1024 * 1024);
             VectorSchemaRoot root =
                     VectorSchemaRoot.create(arrowSchema, inputAllocator)) {
            IntVector ids = (IntVector) root.getVector("id");
            ids.allocateNew(1);
            root.setRowCount(1);

            long inputBytes = inputAllocator.getAllocatedMemory();
            List<ArrowBuf> fieldBuffers = ids.getFieldBuffers();
            int[] refCounts = new int[fieldBuffers.size()];
            for (int i = 0; i < fieldBuffers.size(); i++) {
                refCounts[i] = fieldBuffers.get(i).refCnt();
            }

            ByteArrayOutputStream output = new ByteArrayOutputStream();
            try (MosaicWriter writer =
                    new MosaicWriter(output, arrowSchema, writerAllocator)) {
                long writerBytes = writerAllocator.getAllocatedMemory();
                RuntimeException error =
                        assertThrows(RuntimeException.class, () -> writer.write(root));
                assertTrue(error.getMessage().contains("non-nullable column 'id' has 1 nulls"));
                assertEquals(inputBytes, inputAllocator.getAllocatedMemory());
                assertEquals(writerBytes, writerAllocator.getAllocatedMemory());
                for (int i = 0; i < fieldBuffers.size(); i++) {
                    assertEquals(refCounts[i], fieldBuffers.get(i).refCnt());
                }

                ids.set(0, 7);
                writer.write(root);
                assertEquals(inputBytes, inputAllocator.getAllocatedMemory());
                assertEquals(writerBytes, writerAllocator.getAllocatedMemory());
                for (int i = 0; i < fieldBuffers.size(); i++) {
                    assertEquals(refCounts[i], fieldBuffers.get(i).refCnt());
                }
            }
            data = output.toByteArray();
        }

        int totalRows = 0;
        try (MosaicReader reader = readerFromBytes(data)) {
            for (int rg = 0; rg < reader.numRowGroups(); rg++) {
                try (VectorSchemaRoot batch = reader.readRowGroup(rg, allocator)) {
                    totalRows += batch.getRowCount();
                    assertEquals(7, ((IntVector) batch.getVector("id")).get(0));
                }
            }
        }
        assertEquals(1, totalRows);
    }

    @Test
    public void testOutputWriteFailureAbortsWriterAndPreservesThrowable() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.notNullable("id", new ArrowType.Int(32, true))
        ));
        WriterOptions options =
                new WriterOptions()
                        .compression(0)
                        .numBuckets(1)
                        .rowGroupMaxSize(1);

        try (RootAllocator writerRoot = new RootAllocator(16L * 1024 * 1024);
             RootAllocator inputRoot = new RootAllocator(16L * 1024 * 1024)) {
            try (BufferAllocator writerAllocator =
                         writerRoot.newChildAllocator("writer", 0, 16L * 1024 * 1024);
                 BufferAllocator inputAllocator =
                         inputRoot.newChildAllocator("input", 0, 16L * 1024 * 1024);
                 VectorSchemaRoot root =
                         VectorSchemaRoot.create(arrowSchema, inputAllocator)) {
                IntVector ids = (IntVector) root.getVector("id");
                ids.allocateNew(1);
                ids.set(0, 7);
                root.setRowCount(1);

                long inputBytes = inputAllocator.getAllocatedMemory();
                long writerBytes = writerAllocator.getAllocatedMemory();
                FailOnceOutputStream output =
                        new FailOnceOutputStream(
                                FailurePoint.WRITE, "sentinel-output-write");
                MosaicWriter writer =
                        new MosaicWriter(output, arrowSchema, options, writerAllocator);

                RuntimeException error =
                        assertThrows(RuntimeException.class, () -> writer.write(root));
                assertEquals("write batch failed", error.getMessage());
                assertSame(output.failure, error.getCause());
                assertEquals("sentinel-output-write", error.getCause().getMessage());
                assertSame(output.failureCause, error.getCause().getCause());
                assertEquals(inputBytes, inputAllocator.getAllocatedMemory());
                assertEquals(writerBytes, writerAllocator.getAllocatedMemory());
                assertEquals(1, output.writeCalls);
                assertEquals(0, output.size());

                RuntimeException retryError =
                        assertThrows(RuntimeException.class, () -> writer.write(root));
                assertTrue(retryError
                        .getMessage()
                        .contains("writer is aborted after a previous failure"));
                assertEquals(1, output.writeCalls);

                RuntimeException closeError =
                        assertThrows(RuntimeException.class, writer::close);
                assertTrue(closeError
                        .getMessage()
                        .contains("writer is aborted after a previous failure"));
                assertEquals(1, output.writeCalls);
                assertEquals(0, output.flushCalls);
                assertEquals(0, output.size());
                assertEquals(inputBytes, inputAllocator.getAllocatedMemory());
                assertEquals(writerBytes, writerAllocator.getAllocatedMemory());
            }
            assertEquals(0, inputRoot.getAllocatedMemory());
            assertEquals(0, writerRoot.getAllocatedMemory());
        }
    }

    @Test
    public void testOutputFlushFailurePreservesThrowableWithoutFreeRetry() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.notNullable("id", new ArrowType.Int(32, true))
        ));
        WriterOptions options =
                new WriterOptions()
                        .compression(0)
                        .numBuckets(1)
                        .rowGroupMaxSize(1);

        try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
            IntVector ids = (IntVector) root.getVector("id");
            ids.allocateNew(1);
            ids.set(0, 7);
            root.setRowCount(1);

            FailOnceOutputStream output =
                    new FailOnceOutputStream(
                            FailurePoint.FLUSH, "sentinel-output-flush");
            MosaicWriter writer =
                    new MosaicWriter(output, arrowSchema, options, allocator);
            writer.write(root);
            int writesBeforeClose = output.writeCalls;

            RuntimeException error = assertThrows(RuntimeException.class, writer::close);
            assertEquals("close failed", error.getMessage());
            assertSame(output.failure, error.getCause());
            assertEquals("sentinel-output-flush", error.getCause().getMessage());
            assertSame(output.failureCause, error.getCause().getCause());
            assertTrue(output.writeCalls > writesBeforeClose);
            assertEquals(1, output.flushCalls);
            assertTrue(output.size() > 0);

            int writesAfterClose = output.writeCalls;
            writer.close();
            assertEquals(writesAfterClose, output.writeCalls);
            assertEquals(1, output.flushCalls);
        }
    }

    @Test
    public void testCrossRootPartialExportFailureCanRetryWithoutLeak() {
        byte[] data;
        try (RootAllocator writerRoot = new RootAllocator(16L * 1024 * 1024);
             RootAllocator inputRoot = new RootAllocator(16L * 1024 * 1024)) {
            try (BufferAllocator writerAllocator =
                         writerRoot.newChildAllocator("writer", 0, 16L * 1024 * 1024);
                 BufferAllocator inputAllocator =
                         inputRoot.newChildAllocator("input", 0, 16L * 1024 * 1024)) {
                IntVector first = new IntVector("first", inputAllocator);
                FailOnceListVector second = new FailOnceListVector("second", inputAllocator);
                try (VectorSchemaRoot root = VectorSchemaRoot.of(first, second)) {
                    first.allocateNew(2);
                    second.allocateNew();
                    first.set(0, 1);
                    first.set(1, 2);
                    UnionListWriter listWriter = second.getWriter();
                    listWriter.setPosition(0);
                    listWriter.startList();
                    listWriter.writeInt(3);
                    listWriter.endList();
                    listWriter.setPosition(1);
                    listWriter.startList();
                    listWriter.writeInt(4);
                    listWriter.endList();
                    root.setRowCount(2);

                    long inputBytes = inputRoot.getAllocatedMemory();
                    ByteArrayOutputStream output = new ByteArrayOutputStream();
                    try (MosaicWriter writer =
                            new MosaicWriter(output, root.getSchema(), writerAllocator)) {
                        long writerBytes = writerAllocator.getAllocatedMemory();
                        assertThrows(InjectedExportException.class, () -> writer.write(root));
                        assertEquals(inputBytes, inputRoot.getAllocatedMemory());
                        assertEquals(inputBytes, inputAllocator.getAllocatedMemory());
                        assertEquals(writerBytes, writerAllocator.getAllocatedMemory());
                        assertEquals(2, first.get(1));
                        assertEquals("[4]", second.getObject(1).toString());

                        writer.write(root);
                    }
                    data = output.toByteArray();
                }
            }
            assertEquals(0, inputRoot.getAllocatedMemory());
            assertEquals(0, writerRoot.getAllocatedMemory());
        }

        try (MosaicReader reader = readerFromBytes(data);
             VectorSchemaRoot batch = reader.readRowGroup(0, allocator)) {
            assertEquals(2, batch.getRowCount());
            assertEquals(1, ((IntVector) batch.getVector("first")).get(0));
            assertEquals(2, ((IntVector) batch.getVector("first")).get(1));
            assertEquals("[3]", batch.getVector("second").getObject(0).toString());
            assertEquals("[4]", batch.getVector("second").getObject(1).toString());
        }
    }

    @Test
    public void testWriteSequentialBundlesFromDifferentRootAllocators() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.notNullable("id", new ArrowType.Int(32, true))
        ));

        byte[] data = writeToBytes(arrowSchema, writer -> {
            for (int batch = 0; batch < 2; batch++) {
                try (BufferAllocator inputAllocator = new RootAllocator();
                     VectorSchemaRoot root =
                             VectorSchemaRoot.create(arrowSchema, inputAllocator)) {
                    IntVector ids = (IntVector) root.getVector("id");
                    ids.allocateNew(2);
                    ids.set(0, batch * 2);
                    ids.set(1, batch * 2 + 1);
                    root.setRowCount(2);
                    writer.write(root);
                }
            }
        });

        boolean[] seen = new boolean[4];
        int totalRows = 0;
        try (MosaicReader reader = readerFromBytes(data)) {
            for (int rg = 0; rg < reader.numRowGroups(); rg++) {
                try (VectorSchemaRoot batch = reader.readRowGroup(rg, allocator)) {
                    IntVector ids = (IntVector) batch.getVector("id");
                    for (int i = 0; i < batch.getRowCount(); i++) {
                        seen[ids.get(i)] = true;
                        totalRows++;
                    }
                }
            }
        }
        assertEquals(4, totalRows);
        assertArrayEquals(new boolean[]{true, true, true, true}, seen);
    }

    @Test
    public void testRejectsFieldVectorsFromDifferentAllocatorRoots() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.notNullable("id", new ArrowType.Int(32, true)),
                Field.nullable("name", ArrowType.Utf8.INSTANCE)
        ));

        try (BufferAllocator idAllocator = new RootAllocator();
             BufferAllocator nameAllocator = new RootAllocator()) {
            IntVector ids = new IntVector("id", idAllocator);
            VarCharVector names = new VarCharVector("name", nameAllocator);
            try (VectorSchemaRoot root = VectorSchemaRoot.of(ids, names)) {
                ids.allocateNew(1);
                names.allocateNew(1);
                ids.set(0, 1);
                names.setSafe(0, "one".getBytes());
                root.setRowCount(1);

                IllegalArgumentException error =
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> writeToBytes(arrowSchema, writer -> writer.write(root)));
                assertTrue(error.getMessage().contains("same allocator root"));
                assertTrue(error.getMessage().contains("name"));
            }
        }
    }

    @Test
    public void testRejectsNestedFieldVectorFromDifferentAllocatorRootWithoutLeak() {
        try (RootAllocator parentAllocator = new RootAllocator(16L * 1024 * 1024);
             RootAllocator nestedAllocator = new RootAllocator(16L * 1024 * 1024)) {
            InjectableListVector list = new InjectableListVector("items", parentAllocator);
            IntVector data = new IntVector(ListVector.DATA_VECTOR_NAME, nestedAllocator);
            list.setDataVector(data);

            try (VectorSchemaRoot root = VectorSchemaRoot.of(list)) {
                list.allocateNew();
                list.startNewValue(0);
                data.set(0, 7);
                list.endValue(0, 1);
                list.setValueCount(1);
                root.setRowCount(1);

                long parentBefore = parentAllocator.getAllocatedMemory();
                long nestedBefore = nestedAllocator.getAllocatedMemory();

                IllegalArgumentException error =
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> writeToBytes(root.getSchema(), writer -> writer.write(root)));
                assertTrue(error.getMessage().contains("same allocator root"));
                assertTrue(error.getMessage().contains("items." + ListVector.DATA_VECTOR_NAME));
                assertEquals(parentBefore, parentAllocator.getAllocatedMemory());
                assertEquals(nestedBefore, nestedAllocator.getAllocatedMemory());
            } finally {
                data.close();
            }

            assertEquals(0, parentAllocator.getAllocatedMemory());
            assertEquals(0, nestedAllocator.getAllocatedMemory());
        }
    }

    @Test
    public void testWriterOpenFailurePreservesNativeMessage() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.notNullable("id", new ArrowType.Int(32, true))
        ));
        WriterOptions options = new WriterOptions().statsColumns("missing");

        RuntimeException error =
                assertThrows(
                        RuntimeException.class,
                        () ->
                                new MosaicWriter(
                                        new ByteArrayOutputStream(),
                                        arrowSchema,
                                        options,
                                        allocator));

        assertTrue(
                error.getMessage(),
                error.getMessage()
                        .contains(
                                "writer open failed: stats_columns: column 'missing' not found in schema"));
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
            reader.project(new String[]{"a", "b"});

            int totalRows = 0;
            for (int rg = 0; rg < reader.numRowGroups(); rg++) {
                try (VectorSchemaRoot batch = reader.readRowGroup(rg, allocator)) {
                    totalRows += batch.getRowCount();
                    assertEquals(2, batch.getFieldVectors().size());
                }
            }
            assertEquals(20, totalRows);
        }
    }

    @Test
    public void testProjectionOrder() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("a", new ArrowType.Int(32, true)),
                Field.nullable("b", ArrowType.Utf8.INSTANCE),
                Field.nullable("c", new ArrowType.FloatingPoint(org.apache.arrow.vector.types.FloatingPointPrecision.DOUBLE))
        ));

        byte[] data;
        try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
            IntVector aVec = (IntVector) root.getVector("a");
            VarCharVector bVec = (VarCharVector) root.getVector("b");
            Float8Vector cVec = (Float8Vector) root.getVector("c");

            int n = 10;
            aVec.allocateNew(n);
            bVec.allocateNew(n);
            cVec.allocateNew(n);
            for (int i = 0; i < n; i++) {
                aVec.set(i, i);
                bVec.setSafe(i, ("s" + i).getBytes());
                cVec.set(i, i * 0.5);
            }
            root.setRowCount(n);
            data = writeToBytes(arrowSchema, new WriterOptions().numBuckets(2), writer -> writer.write(root));
        }

        try (MosaicReader reader = readerFromBytes(data)) {
            reader.project(new String[]{"c", "a", "b"});
            try (VectorSchemaRoot batch = reader.readRowGroup(0, allocator)) {
                assertEquals(3, batch.getFieldVectors().size());
                assertEquals("c", batch.getVector(0).getName());
                assertEquals("a", batch.getVector(1).getName());
                assertEquals("b", batch.getVector(2).getName());
                assertEquals(10, batch.getRowCount());

                Float8Vector cOut = (Float8Vector) batch.getVector(0);
                IntVector aOut = (IntVector) batch.getVector(1);
                VarCharVector bOut = (VarCharVector) batch.getVector(2);
                for (int i = 0; i < 10; i++) {
                    assertEquals(i, aOut.get(i));
                    assertEquals("s" + i, new String(bOut.get(i)));
                    assertEquals(i * 0.5, cOut.get(i), 1e-10);
                }
            }
        }
    }

    @Test
    public void testProjectionEmpty() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("a", new ArrowType.Int(32, true)),
                Field.nullable("b", ArrowType.Utf8.INSTANCE)
        ));

        byte[] data;
        try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
            IntVector aVec = (IntVector) root.getVector("a");
            VarCharVector bVec = (VarCharVector) root.getVector("b");

            int n = 5;
            aVec.allocateNew(n);
            bVec.allocateNew(n);
            for (int i = 0; i < n; i++) {
                aVec.set(i, i);
                bVec.setSafe(i, ("v" + i).getBytes());
            }
            root.setRowCount(n);
            data = writeToBytes(arrowSchema, writer -> writer.write(root));
        }

        try (MosaicReader reader = readerFromBytes(data)) {
            reader.project(new String[]{});
            try (VectorSchemaRoot batch = reader.readRowGroup(0, allocator)) {
                assertEquals(0, batch.getFieldVectors().size());
                assertEquals(5, batch.getRowCount());
            }
        }
    }

    @Test
    public void testStats() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("id", new ArrowType.Int(32, true)),
                Field.nullable("name", ArrowType.Utf8.INSTANCE),
                Field.nullable("value", new ArrowType.FloatingPoint(org.apache.arrow.vector.types.FloatingPointPrecision.DOUBLE))
        ));

        WriterOptions opts = new WriterOptions().statsColumns("id", "value");

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
                java.util.Map<String, ColumnStatistics> stats = reader.getRowGroupStatistics(rg);
                assertTrue(stats.size() > 0);
                assertTrue(stats.containsKey("id"));
                assertTrue(stats.containsKey("value"));
                for (ColumnStatistics stat : stats.values()) {
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
    public void testTimestampNsRoundtrip() {
        ArrowType.Timestamp tsNsType = new ArrowType.Timestamp(TimeUnit.NANOSECOND, null);
        ArrowType.Timestamp tsNsTzType = new ArrowType.Timestamp(TimeUnit.NANOSECOND, "Asia/Shanghai");
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("ts_ns", tsNsType),
                Field.nullable("ts_ns_tz", tsNsTzType)
        ));

        long[] values = {1700000000000000123L, -1L};
        byte[] data;
        try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
            TimeStampNanoVector tsNsVec = (TimeStampNanoVector) root.getVector("ts_ns");
            TimeStampNanoTZVector tsNsTzVec = (TimeStampNanoTZVector) root.getVector("ts_ns_tz");
            int n = 3;
            tsNsVec.allocateNew(n);
            tsNsTzVec.allocateNew(n);

            tsNsVec.set(0, values[0]);
            tsNsVec.setNull(1);
            tsNsVec.set(2, values[1]);
            tsNsTzVec.set(0, values[0]);
            tsNsTzVec.setNull(1);
            tsNsTzVec.set(2, values[1]);

            root.setRowCount(n);
            data = writeToBytes(arrowSchema, writer -> writer.write(root));
        }

        try (MosaicReader reader = readerFromBytes(data)) {
            assertEquals(tsNsType, reader.getSchema().findField("ts_ns").getType());
            assertEquals(tsNsTzType, reader.getSchema().findField("ts_ns_tz").getType());
            try (VectorSchemaRoot batch = reader.readRowGroup(0, allocator)) {
                TimeStampNanoVector tsNs = (TimeStampNanoVector) batch.getVector("ts_ns");
                TimeStampNanoTZVector tsNsTz = (TimeStampNanoTZVector) batch.getVector("ts_ns_tz");

                assertEquals(values[0], tsNs.get(0));
                assertTrue(tsNs.isNull(1));
                assertEquals(values[1], tsNs.get(2));
                assertEquals(values[0], tsNsTz.get(0));
                assertTrue(tsNsTz.isNull(1));
                assertEquals(values[1], tsNsTz.get(2));
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
    public void testReaderOpenFreesNativeHandleWhenConstructorFails() throws Exception {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("x", new ArrowType.Int(32, true))
        ));

        byte[] data;
        try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
            IntVector xVec = (IntVector) root.getVector("x");
            xVec.allocateNew(1);
            xVec.set(0, 1);
            root.setRowCount(1);
            data = writeToBytes(arrowSchema, writer -> writer.write(root));
        }

        WeakReference<InputFile> reference = openReaderWithClosedAllocator(data);
        awaitGarbageCollection(reference);
    }

    @Test
    public void testReaderOpenReleasesInputGlobalRefWhenReadFails() throws Exception {
        awaitGarbageCollection(openReaderWithFailingInput());
    }

    @Test
    public void testReaderRestoresBackgroundInputExceptionAndReleasesGlobalRef()
            throws Exception {
        Schema schema = new Schema(Arrays.asList(
                Field.nullable("value", new ArrowType.Int(32, true))
        ));
        byte[] data;
        try (VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
            IntVector values = (IntVector) root.getVector("value");
            values.allocateNew(1);
            values.set(0, 7);
            root.setRowCount(1);
            data = writeToBytes(schema, writer -> writer.write(root));
        }

        awaitGarbageCollection(readRowGroupWithFailingInput(data));
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

        WriterOptions opts = new WriterOptions().statsColumns("a", "b").numBuckets(1);

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
            java.util.Map<String, ColumnStatistics> stats = reader.getRowGroupStatistics(0);
            assertEquals(2, stats.size());

            ColumnStatistics aStat = stats.get("a");
            assertEquals(1, aStat.getNullCount());
            assertTrue(aStat.hasMinMax());
            int minA = ByteBuffer.wrap(aStat.getMin()).order(ByteOrder.BIG_ENDIAN).getInt();
            int maxA = ByteBuffer.wrap(aStat.getMax()).order(ByteOrder.BIG_ENDIAN).getInt();
            assertEquals(5, minA);
            assertEquals(20, maxA);

            ColumnStatistics bStat = stats.get("b");
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

        WriterOptions opts = new WriterOptions().statsColumns("x").numBuckets(1);

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
            java.util.Map<String, ColumnStatistics> stats = reader.getRowGroupStatistics(0);
            assertEquals(1, stats.size());
            ColumnStatistics xStat = stats.get("x");
            assertEquals(3, xStat.getNullCount());
            assertFalse(xStat.hasMinMax());
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
                Field.nullable("name", ArrowType.Utf8.INSTANCE),
                Field.notNullable("id", new ArrowType.Int(32, true)),
                Field.nullable("score", new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE))
        ));

        byte[] data;
        try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
            VarCharVector names = (VarCharVector) root.getVector("name");
            IntVector ids = (IntVector) root.getVector("id");
            Float8Vector scores = (Float8Vector) root.getVector("score");
            names.allocateNew(1); ids.allocateNew(1); scores.allocateNew(1);
            names.setSafe(0, "x".getBytes()); ids.set(0, 1); scores.set(0, 1.0);
            root.setRowCount(1);
            data = writeToBytes(arrowSchema, writer -> writer.write(root));
        }

        try (MosaicReader reader = readerFromBytes(data)) {
            Schema readSchema = reader.getSchema();
            assertEquals(3, readSchema.getFields().size());
            assertEquals("name", readSchema.getFields().get(0).getName());
            assertEquals("id", readSchema.getFields().get(1).getName());
            assertEquals("score", readSchema.getFields().get(2).getName());
            assertFalse(readSchema.getFields().get(1).isNullable());
            assertTrue(readSchema.getFields().get(0).isNullable());
        }
    }

    @Test
    public void testWriterStats() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("id", new ArrowType.Int(32, true)),
                Field.nullable("name", ArrowType.Utf8.INSTANCE),
                Field.nullable("score", new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE))
        ));

        WriterOptions opts = new WriterOptions().statsColumns("id", "score");

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        MosaicWriter writer = new MosaicWriter(baos, arrowSchema, opts, allocator);
        try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
            IntVector ids = (IntVector) root.getVector("id");
            VarCharVector names = (VarCharVector) root.getVector("name");
            Float8Vector scores = (Float8Vector) root.getVector("score");

            int n = 10;
            ids.allocateNew(n);
            names.allocateNew(n);
            scores.allocateNew(n);

            for (int i = 0; i < n; i++) {
                ids.set(i, i * 10);
                names.setSafe(i, ("item_" + i).getBytes());
                scores.set(i, i * 1.1);
            }
            root.setRowCount(n);
            writer.write(root);
        }
        writer.close();

        assertEquals(1, writer.numRowGroups());
        java.util.Map<String, ColumnStatistics> stats = writer.getRowGroupStatistics(0);
        assertTrue(stats.size() > 0);
        assertTrue(stats.containsKey("id"));
        assertTrue(stats.containsKey("score"));
        for (ColumnStatistics stat : stats.values()) {
            assertEquals(0, stat.getNullCount());
            assertTrue(stat.hasMinMax());
            assertNotNull(stat.getMin());
            assertNotNull(stat.getMax());
        }

        ColumnStatistics idStat = stats.get("id");
        int minId = ByteBuffer.wrap(idStat.getMin()).order(ByteOrder.BIG_ENDIAN).getInt();
        int maxId = ByteBuffer.wrap(idStat.getMax()).order(ByteOrder.BIG_ENDIAN).getInt();
        assertEquals(0, minId);
        assertEquals(90, maxId);
    }

    @Test
    public void testWriterStatsWithNulls() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("a", new ArrowType.Int(32, true)),
                Field.nullable("b", new ArrowType.Int(64, true))
        ));

        WriterOptions opts = new WriterOptions().statsColumns("a", "b").numBuckets(1);

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        MosaicWriter writer = new MosaicWriter(baos, arrowSchema, opts, allocator);
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
            writer.write(root);
        }
        writer.close();

        assertEquals(1, writer.numRowGroups());
        java.util.Map<String, ColumnStatistics> stats = writer.getRowGroupStatistics(0);
        assertEquals(2, stats.size());

        ColumnStatistics aStat = stats.get("a");
        assertEquals(1, aStat.getNullCount());
        assertTrue(aStat.hasMinMax());
        int minA = ByteBuffer.wrap(aStat.getMin()).order(ByteOrder.BIG_ENDIAN).getInt();
        int maxA = ByteBuffer.wrap(aStat.getMax()).order(ByteOrder.BIG_ENDIAN).getInt();
        assertEquals(5, minA);
        assertEquals(20, maxA);

        ColumnStatistics bStat = stats.get("b");
        assertEquals(2, bStat.getNullCount());
        assertTrue(bStat.hasMinMax());
        long minB = ByteBuffer.wrap(bStat.getMin()).order(ByteOrder.BIG_ENDIAN).getLong();
        long maxB = ByteBuffer.wrap(bStat.getMax()).order(ByteOrder.BIG_ENDIAN).getLong();
        assertEquals(50, minB);
        assertEquals(100, maxB);
    }

    @Test
    public void testWriterStatsAllNull() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("x", new ArrowType.Int(32, true))
        ));

        WriterOptions opts = new WriterOptions().statsColumns("x").numBuckets(1);

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        MosaicWriter writer = new MosaicWriter(baos, arrowSchema, opts, allocator);
        try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
            IntVector xVec = (IntVector) root.getVector("x");
            xVec.allocateNew(3);
            xVec.setNull(0);
            xVec.setNull(1);
            xVec.setNull(2);
            root.setRowCount(3);
            writer.write(root);
        }
        writer.close();

        assertEquals(1, writer.numRowGroups());
        java.util.Map<String, ColumnStatistics> stats = writer.getRowGroupStatistics(0);
        assertEquals(1, stats.size());
        ColumnStatistics xStat = stats.get("x");
        assertEquals(3, xStat.getNullCount());
        assertFalse(xStat.hasMinMax());
    }

    @Test
    public void testWriterStatsMatchesReaderStats() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("id", new ArrowType.Int(32, true)),
                Field.nullable("value", new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE))
        ));

        WriterOptions opts = new WriterOptions().statsColumns("id", "value").numBuckets(1);

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        MosaicWriter writer = new MosaicWriter(baos, arrowSchema, opts, allocator);
        try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
            IntVector ids = (IntVector) root.getVector("id");
            Float8Vector values = (Float8Vector) root.getVector("value");
            int n = 20;
            ids.allocateNew(n);
            values.allocateNew(n);
            for (int i = 0; i < n; i++) {
                ids.set(i, i * 5);
                values.set(i, i * 2.5);
            }
            root.setRowCount(n);
            writer.write(root);
        }
        writer.close();

        byte[] data = baos.toByteArray();
        try (MosaicReader reader = readerFromBytes(data)) {
            java.util.Map<String, ColumnStatistics> writerStats = writer.getRowGroupStatistics(0);
            java.util.Map<String, ColumnStatistics> readerStats = reader.getRowGroupStatistics(0);

            assertEquals(writerStats.size(), readerStats.size());
            for (String colName : writerStats.keySet()) {
                ColumnStatistics ws = writerStats.get(colName);
                ColumnStatistics rs = readerStats.get(colName);
                assertNotNull(rs);
                assertEquals(ws.getNullCount(), rs.getNullCount());
                assertEquals(ws.hasMinMax(), rs.hasMinMax());
                assertArrayEquals(ws.getMin(), rs.getMin());
                assertArrayEquals(ws.getMax(), rs.getMax());
            }
        }
    }

    @Test
    public void testRowGroupNumRows() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("id", new ArrowType.Int(32, true)),
                Field.nullable("data", new ArrowType.Int(64, true))
        ));

        WriterOptions opts = new WriterOptions().compression(0).numBuckets(1).rowGroupMaxSize(200);

        int totalRows = 500;
        int batchSize = 10;
        byte[] data = writeToBytes(arrowSchema, opts, writer -> {
            for (int start = 0; start < totalRows; start += batchSize) {
                try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
                    IntVector idVec = (IntVector) root.getVector("id");
                    BigIntVector dataVec = (BigIntVector) root.getVector("data");
                    idVec.allocateNew(batchSize);
                    dataVec.allocateNew(batchSize);
                    for (int i = 0; i < batchSize; i++) {
                        idVec.set(i, start + i);
                        dataVec.set(i, (long) (start + i) * 2);
                    }
                    root.setRowCount(batchSize);
                    writer.write(root);
                }
            }
        });

        try (MosaicReader reader = readerFromBytes(data)) {
            assertTrue(reader.numRowGroups() > 1);
            int sum = 0;
            for (int rg = 0; rg < reader.numRowGroups(); rg++) {
                int numRows = reader.rowGroupNumRows(rg);
                assertTrue(numRows > 0);
                try (VectorSchemaRoot batch = reader.readRowGroup(rg, allocator)) {
                    assertEquals(numRows, batch.getRowCount());
                }
                sum += numRows;
            }
            assertEquals(totalRows, sum);
        }
    }

    @Test
    public void testStatsEmptyStringMin() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("s", ArrowType.Utf8.INSTANCE)
        ));

        WriterOptions opts = new WriterOptions().statsColumns("s").numBuckets(1);

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        MosaicWriter writer = new MosaicWriter(baos, arrowSchema, opts, allocator);
        try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
            VarCharVector sVec = (VarCharVector) root.getVector("s");
            sVec.allocateNew(2);
            sVec.setSafe(0, "".getBytes());
            sVec.setSafe(1, "b".getBytes());
            root.setRowCount(2);
            writer.write(root);
        }
        writer.close();

        // Writer stats: empty string min should still report hasMinMax
        assertEquals(1, writer.numRowGroups());
        java.util.Map<String, ColumnStatistics> writerStats = writer.getRowGroupStatistics(0);
        assertEquals(1, writerStats.size());
        ColumnStatistics wStat = writerStats.get("s");
        assertTrue(wStat.hasMinMax());
        assertArrayEquals(new byte[0], wStat.getMin());
        assertArrayEquals("b".getBytes(), wStat.getMax());
        assertEquals(0, wStat.getNullCount());

        // Reader stats: same assertions
        byte[] data = baos.toByteArray();
        try (MosaicReader reader = readerFromBytes(data)) {
            java.util.Map<String, ColumnStatistics> readerStats = reader.getRowGroupStatistics(0);
            assertEquals(1, readerStats.size());
            ColumnStatistics rStat = readerStats.get("s");
            assertTrue(rStat.hasMinMax());
            assertArrayEquals(new byte[0], rStat.getMin());
            assertArrayEquals("b".getBytes(), rStat.getMax());
            assertEquals(0, rStat.getNullCount());
        }
    }

    @Test
    public void testRowGroupNumRowsSingleRowGroup() {
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("x", new ArrowType.Int(32, true))
        ));

        byte[] data;
        try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
            IntVector xVec = (IntVector) root.getVector("x");
            xVec.allocateNew(10);
            for (int i = 0; i < 10; i++) {
                xVec.set(i, i);
            }
            root.setRowCount(10);
            data = writeToBytes(arrowSchema, writer -> writer.write(root));
        }

        try (MosaicReader reader = readerFromBytes(data)) {
            assertEquals(1, reader.numRowGroups());
            assertEquals(10, reader.rowGroupNumRows(0));
        }
    }

    @Test
    public void testArrayType() {
        Field elementField = new Field("item", FieldType.nullable(new ArrowType.Int(32, true)), null);
        Field listField = new Field("tags", FieldType.nullable(ArrowType.List.INSTANCE), Arrays.asList(elementField));
        Schema arrowSchema = new Schema(Arrays.asList(
                Field.nullable("id", new ArrowType.Int(32, true)),
                listField
        ));

        byte[] data;
        try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
            IntVector ids = (IntVector) root.getVector("id");
            ListVector tags = (ListVector) root.getVector("tags");

            ids.allocateNew(4);
            tags.allocateNew();

            UnionListWriter listWriter = tags.getWriter();

            // Row 0: [10, 20, 30]
            ids.set(0, 1);
            listWriter.setPosition(0);
            listWriter.startList();
            listWriter.writeInt(10);
            listWriter.writeInt(20);
            listWriter.writeInt(30);
            listWriter.endList();

            // Row 1: [40, 50]
            ids.set(1, 2);
            listWriter.setPosition(1);
            listWriter.startList();
            listWriter.writeInt(40);
            listWriter.writeInt(50);
            listWriter.endList();

            // Row 2: [] (empty array)
            ids.set(2, 3);
            listWriter.setPosition(2);
            listWriter.startList();
            listWriter.endList();

            // Row 3: null
            ids.set(3, 4);
            tags.setNull(3);

            root.setRowCount(4);
            data = writeToBytes(arrowSchema, writer -> writer.write(root));
        }

        try (MosaicReader reader = readerFromBytes(data)) {
            try (VectorSchemaRoot batch = reader.readRowGroup(0, allocator)) {
                assertEquals(4, batch.getRowCount());

                IntVector readIds = (IntVector) batch.getVector("id");
                ListVector readTags = (ListVector) batch.getVector("tags");

                assertEquals(1, readIds.get(0));
                assertEquals(2, readIds.get(1));
                assertEquals(3, readIds.get(2));
                assertEquals(4, readIds.get(3));

                // Row 0: [10, 20, 30]
                assertFalse(readTags.isNull(0));
                java.util.List<?> row0 = readTags.getObject(0);
                assertEquals(3, row0.size());
                assertEquals(10, row0.get(0));
                assertEquals(20, row0.get(1));
                assertEquals(30, row0.get(2));

                // Row 1: [40, 50]
                assertFalse(readTags.isNull(1));
                java.util.List<?> row1 = readTags.getObject(1);
                assertEquals(2, row1.size());
                assertEquals(40, row1.get(0));
                assertEquals(50, row1.get(1));

                // Row 2: []
                assertFalse(readTags.isNull(2));
                java.util.List<?> row2 = readTags.getObject(2);
                assertEquals(0, row2.size());

                // Row 3: null
                assertTrue(readTags.isNull(3));
            }
        }
    }

    @Test
    public void testMapType() {
        // Use MapVector's writer to avoid schema mismatch with UnionMapWriter
        Field keyField = new Field("keys", FieldType.notNullable(new ArrowType.Int(32, true)), null);
        Field valueField = new Field("values", FieldType.nullable(ArrowType.Utf8.INSTANCE), null);
        Field entriesField = new Field("entries",
            new FieldType(false, ArrowType.Struct.INSTANCE, null),
            Arrays.asList(keyField, valueField));
        Field mapField = new Field("props",
            new FieldType(true, new ArrowType.Map(false), null),
            Arrays.asList(entriesField));

        Schema arrowSchema = new Schema(Arrays.asList(
                Field.notNullable("id", new ArrowType.Int(32, true)),
                mapField
        ));

        byte[] data;
        try (VectorSchemaRoot root = VectorSchemaRoot.create(arrowSchema, allocator)) {
            IntVector ids = (IntVector) root.getVector("id");
            MapVector mapVec = (MapVector) root.getVector("props");

            ids.allocateNew(3);

            IntVector keyVec = (IntVector) mapVec.getDataVector().getChildrenFromFields().get(0);
            VarCharVector valVec = (VarCharVector) mapVec.getDataVector().getChildrenFromFields().get(1);

            // Row 0: {1: "a", 2: "b"} -> offsets [0, 2]
            mapVec.startNewValue(0);
            keyVec.setSafe(0, 1);
            valVec.setSafe(0, "a".getBytes());
            keyVec.setSafe(1, 2);
            valVec.setSafe(1, "b".getBytes());
            mapVec.endValue(0, 2);

            // Row 1: null
            mapVec.setNull(1);

            // Row 2: {} -> offsets [2, 2]
            mapVec.startNewValue(2);
            mapVec.endValue(2, 0);

            ids.set(0, 1);
            ids.set(1, 2);
            ids.set(2, 3);

            root.setRowCount(3);
            data = writeToBytes(arrowSchema, writer -> writer.write(root));
        }

        try (MosaicReader reader = readerFromBytes(data)) {
            try (VectorSchemaRoot batch = reader.readRowGroup(0, allocator)) {
                assertEquals(3, batch.getRowCount());

                MapVector readMap = (MapVector) batch.getVector("props");
                assertFalse(readMap.isNull(0));
                assertTrue(readMap.isNull(1));
                assertFalse(readMap.isNull(2));

                // Row 0: 2 entries
                java.util.List<?> row0 = readMap.getObject(0);
                assertEquals(2, row0.size());

                // Row 2: empty
                java.util.List<?> row2 = readMap.getObject(2);
                assertEquals(0, row2.size());
            }
        }
    }
}
