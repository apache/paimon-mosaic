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

import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.c.jni.JniWrapper;
import org.apache.arrow.c.jni.PrivateData;
import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Schema;

public class MosaicWriter implements AutoCloseable {

    private long handle;
    private boolean closed;
    private final BufferAllocator allocator;
    private List<Map<String, ColumnStatistics>> rowGroupStats;

    public MosaicWriter(OutputStream outputStream, Schema arrowSchema, BufferAllocator allocator) {
        this(outputStream, arrowSchema, new WriterOptions(), allocator);
    }

    public MosaicWriter(OutputStream outputStream, Schema arrowSchema, WriterOptions options, BufferAllocator allocator) {
        this.allocator = allocator;
        try (ArrowSchema cSchema = ArrowSchema.allocateNew(allocator)) {
            try {
                Data.exportSchema(allocator, arrowSchema, null, cSchema);
                this.handle = NativeLib.nativeWriterOpen(
                        outputStream,
                        cSchema.memoryAddress(),
                        options.getNumBuckets(),
                        options.getCompression(),
                        options.getZstdLevel(),
                        options.getRowGroupMaxSize(),
                        options.getMaxDictTotalBytes(),
                        options.getMaxDictEntries(),
                        options.getStatsColumns(),
                        options.getPageSizeThreshold());
            } finally {
                releaseExported(cSchema);
            }
        }
        if (this.handle == 0) {
            throw new RuntimeException("failed to open writer");
        }
    }

    /**
     * Writes an Arrow batch synchronously.
     *
     * <p>All top-level and nested field vectors must share one allocator root. That root may be
     * independent from the writer allocator root; temporary Arrow C Data metadata remains charged
     * to the writer allocator supplied at construction time. The caller retains ownership of the
     * batch and must keep it open until this method returns.
     *
     * @param root batch to write
     * @throws IllegalArgumentException if field vectors use different allocator roots
     * @throws IllegalStateException if the writer is closed
     */
    public void write(VectorSchemaRoot root) {
        if (closed || handle == 0) {
            throw new IllegalStateException("writer is closed");
        }
        boolean sameAllocatorRoot = sharesWriterAllocatorRoot(root);
        try (ArrowArray arrowArray = ArrowArray.allocateNew(allocator);
             ArrowSchema arrowSchema = ArrowSchema.allocateNew(allocator)) {
            try {
                if (sameAllocatorRoot) {
                    Data.exportVectorSchemaRoot(
                            allocator, root, null, arrowArray, arrowSchema);
                } else {
                    Data.exportSchema(allocator, root.getSchema(), null, arrowSchema);
                    exportCrossRootArray(allocator, root, arrowArray);
                }
                NativeLib.nativeWriterWriteBatch(handle, arrowArray.memoryAddress(), arrowSchema.memoryAddress());
            } finally {
                releaseExported(arrowArray);
                releaseExported(arrowSchema);
            }
        }
    }

    private static void exportCrossRootArray(
            BufferAllocator exportAllocator, VectorSchemaRoot root, ArrowArray arrowArray) {
        // Data.exportVectorSchemaRoot reloads every field into a temporary StructVector. If that
        // reload fails before the root ArrowArray owns a release callback, Arrow 15 can retain
        // already-associated input buffers. Export each child directly instead: input buffers are
        // retained without being associated with the writer allocator, while temporary C Data
        // metadata remains charged to that allocator.
        RootArrayPrivateData privateData = new RootArrayPrivateData();
        try {
            privateData.bufferPointers = exportAllocator.buffer(Long.BYTES);
            privateData.bufferPointers.writeLong(0L);

            List<FieldVector> vectors = root.getFieldVectors();
            if (!vectors.isEmpty()) {
                privateData.childPointers =
                        exportAllocator.buffer((long) vectors.size() * Long.BYTES);
                for (int i = 0; i < vectors.size(); i++) {
                    ArrowArray child = ArrowArray.allocateNew(exportAllocator);
                    privateData.children.add(child);
                    privateData.childPointers.writeLong(child.memoryAddress());
                }
                for (int i = 0; i < vectors.size(); i++) {
                    Data.exportVector(
                            exportAllocator, vectors.get(i), null, privateData.children.get(i));
                }
            }

            ArrowArray.Snapshot snapshot = new ArrowArray.Snapshot();
            snapshot.length = root.getRowCount();
            snapshot.null_count = 0;
            snapshot.offset = 0;
            snapshot.n_buffers = 1;
            snapshot.n_children = vectors.size();
            snapshot.buffers = privateData.bufferPointers.memoryAddress();
            snapshot.children =
                    privateData.childPointers == null
                            ? 0
                            : privateData.childPointers.memoryAddress();
            snapshot.dictionary = 0;
            snapshot.release = 0;
            arrowArray.save(snapshot);
            JniWrapper.get().exportArray(arrowArray.memoryAddress(), privateData);
        } catch (RuntimeException | Error failure) {
            privateData.abort(failure);
            throw failure;
        }
    }

    private static final class RootArrayPrivateData implements PrivateData {

        private ArrowBuf bufferPointers;
        private ArrowBuf childPointers;
        private final List<ArrowArray> children = new ArrayList<>();

        private void abort(Throwable failure) {
            for (ArrowArray child : children) {
                try {
                    releaseExported(child);
                } catch (RuntimeException | Error cleanupFailure) {
                    failure.addSuppressed(cleanupFailure);
                }
            }
            try {
                close();
            } catch (RuntimeException | Error cleanupFailure) {
                failure.addSuppressed(cleanupFailure);
            }
        }

        @Override
        public void close() {
            if (bufferPointers != null) {
                bufferPointers.close();
                bufferPointers = null;
            }
            if (childPointers != null) {
                childPointers.close();
                childPointers = null;
            }
            for (ArrowArray child : children) {
                child.close();
            }
            children.clear();
        }
    }

    private boolean sharesWriterAllocatorRoot(VectorSchemaRoot root) {
        List<FieldVector> vectors = root.getFieldVectors();
        if (vectors.isEmpty()) {
            return true;
        }

        BufferAllocator inputRoot = vectors.get(0).getAllocator().getRoot();
        for (FieldVector vector : vectors) {
            validateAllocatorRoot(vector, inputRoot, vector.getField().getName());
        }
        return inputRoot == allocator.getRoot();
    }

    private static void validateAllocatorRoot(
            FieldVector vector, BufferAllocator expectedRoot, String fieldPath) {
        if (vector.getAllocator().getRoot() != expectedRoot) {
            throw new IllegalArgumentException(
                    "All field vectors must share the same allocator root; field '"
                            + fieldPath
                            + "' uses a different root");
        }
        for (FieldVector child : vector.getChildrenFromFields()) {
            validateAllocatorRoot(
                    child, expectedRoot, fieldPath + "." + child.getField().getName());
        }
    }

    private static void releaseExported(ArrowSchema schema) {
        if (schema.snapshot().release != 0) {
            schema.release();
        }
    }

    private static void releaseExported(ArrowArray array) {
        if (array.snapshot().release != 0) {
            array.release();
        }
    }

    public long estimatedFileSize() {
        return NativeLib.nativeWriterEstimatedSize(handle);
    }

    public int numRowGroups() {
        if (rowGroupStats == null) {
            throw new IllegalStateException("writer is not closed yet");
        }
        return rowGroupStats.size();
    }

    /**
     * Returns column statistics for the given row group, keyed by column name.
     */
    public Map<String, ColumnStatistics> getRowGroupStatistics(int rgIndex) {
        if (rowGroupStats == null) {
            throw new IllegalStateException("writer is not closed yet");
        }
        return rowGroupStats.get(rgIndex);
    }

    @Override
    public void close() {
        if (!closed && handle != 0) {
            closed = true;
            try {
                NativeLib.nativeWriterClose(handle);
                collectStatistics();
            } finally {
                NativeLib.nativeWriterFree(handle);
                handle = 0;
            }
        }
    }

    private void collectStatistics() {
        int numRg = NativeLib.nativeWriterNumRowGroups(handle);
        List<Map<String, ColumnStatistics>> allStats = new ArrayList<>(numRg);
        for (int rg = 0; rg < numRg; rg++) {
            String[] names = NativeLib.nativeWriterRowGroupStatNames(handle, rg);
            if (names == null || names.length == 0) {
                allStats.add(Collections.emptyMap());
                continue;
            }
            long[] nullCounts = NativeLib.nativeWriterRowGroupStatNullCounts(handle, rg);
            byte[][] mins = NativeLib.nativeWriterRowGroupStatMins(handle, rg);
            byte[][] maxs = NativeLib.nativeWriterRowGroupStatMaxs(handle, rg);
            Map<String, ColumnStatistics> rgStats = new LinkedHashMap<>(names.length);
            for (int i = 0; i < names.length; i++) {
                rgStats.put(names[i], new ColumnStatistics(nullCounts[i], mins[i], maxs[i]));
            }
            allStats.add(Collections.unmodifiableMap(rgStats));
        }
        this.rowGroupStats = Collections.unmodifiableList(allStats);
    }
}
