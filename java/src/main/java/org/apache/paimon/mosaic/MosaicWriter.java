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

import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Schema;

public class MosaicWriter implements AutoCloseable {

    private long handle;
    private boolean closed;
    private final BufferAllocator allocator;

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

    public void write(VectorSchemaRoot root) {
        if (closed || handle == 0) {
            throw new IllegalStateException("writer is closed");
        }
        try (ArrowArray arrowArray = ArrowArray.allocateNew(allocator);
             ArrowSchema arrowSchema = ArrowSchema.allocateNew(allocator)) {
            try {
                Data.exportVectorSchemaRoot(allocator, root, null, arrowArray, arrowSchema);
                NativeLib.nativeWriterWriteBatch(handle, arrowArray.memoryAddress(), arrowSchema.memoryAddress());
            } finally {
                releaseExported(arrowArray);
                releaseExported(arrowSchema);
            }
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

    @Override
    public void close() {
        if (!closed && handle != 0) {
            closed = true;
            try {
                NativeLib.nativeWriterClose(handle);
            } finally {
                NativeLib.nativeWriterFree(handle);
                handle = 0;
            }
        }
    }
}
