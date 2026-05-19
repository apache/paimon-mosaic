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

package io.mosaic;

import java.util.ArrayList;
import java.util.List;

import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Schema;

public class MosaicReader implements AutoCloseable {

    private long handle;
    private final Schema schema;

    private MosaicReader(long handle, BufferAllocator allocator) {
        this.handle = handle;
        try (ArrowSchema cSchema = ArrowSchema.allocateNew(allocator)) {
            int rc = NativeLib.nativeReaderExportSchema(handle, cSchema.memoryAddress());
            if (rc != 0) {
                throw new RuntimeException("failed to export schema");
            }
            this.schema = Data.importSchema(allocator, cSchema, null);
        }
    }

    public static MosaicReader open(InputFile inputFile, long fileLength, BufferAllocator allocator) {
        long h = NativeLib.nativeReaderOpen(inputFile, fileLength);
        if (h == 0) {
            throw new RuntimeException("failed to open reader");
        }
        return new MosaicReader(h, allocator);
    }

    public Schema getSchema() {
        return schema;
    }

    public int numRowGroups() {
        return NativeLib.nativeReaderNumRowGroups(handle);
    }

    public VectorSchemaRoot readRowGroup(int rgIndex, BufferAllocator allocator) {
        long rgHandle = NativeLib.nativeReaderOpenRowGroup(handle, rgIndex);
        if (rgHandle == 0) {
            throw new RuntimeException("failed to open row group " + rgIndex);
        }
        try {
            return readRowGroupHandle(rgHandle, allocator);
        } finally {
            NativeLib.nativeRowGroupReaderFree(rgHandle);
        }
    }

    public VectorSchemaRoot readRowGroup(int rgIndex, int[] columns, BufferAllocator allocator) {
        long rgHandle = NativeLib.nativeReaderOpenRowGroupProjected(handle, rgIndex, columns);
        if (rgHandle == 0) {
            throw new RuntimeException("failed to open row group " + rgIndex);
        }
        try {
            return readRowGroupHandle(rgHandle, allocator);
        } finally {
            NativeLib.nativeRowGroupReaderFree(rgHandle);
        }
    }

    private VectorSchemaRoot readRowGroupHandle(long rgHandle, BufferAllocator allocator) {
        try (ArrowArray arrowArray = ArrowArray.allocateNew(allocator);
             ArrowSchema arrowSchema = ArrowSchema.allocateNew(allocator)) {
            int rc = NativeLib.nativeRowGroupReaderReadColumns(
                    rgHandle, arrowArray.memoryAddress(), arrowSchema.memoryAddress());
            if (rc != 0) {
                throw new RuntimeException("readColumns failed");
            }
            return Data.importVectorSchemaRoot(allocator, arrowArray, arrowSchema, null);
        }
    }

    public List<ColumnStatistics> getRowGroupStatistics(int rgIndex) {
        int n = NativeLib.nativeReaderRowGroupNumStats(handle, rgIndex);
        if (n < 0) {
            throw new RuntimeException("failed to get row group statistics for index " + rgIndex);
        }
        List<ColumnStatistics> result = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            result.add(new ColumnStatistics(
                    NativeLib.nativeReaderRowGroupStatColumnIndex(handle, rgIndex, i),
                    NativeLib.nativeReaderRowGroupStatNullCount(handle, rgIndex, i),
                    NativeLib.nativeReaderRowGroupStatMin(handle, rgIndex, i),
                    NativeLib.nativeReaderRowGroupStatMax(handle, rgIndex, i)));
        }
        return result;
    }

    @Override
    public void close() {
        if (handle != 0) {
            NativeLib.nativeReaderFree(handle);
            handle = 0;
        }
    }
}
