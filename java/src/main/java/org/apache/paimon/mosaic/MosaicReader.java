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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
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
        long handle = NativeLib.nativeReaderOpen(inputFile, fileLength);
        if (handle == 0) {
            throw new RuntimeException("failed to open reader");
        }
        try {
            return new MosaicReader(handle, allocator);
        } catch (RuntimeException | Error e) {
            NativeLib.nativeReaderFree(handle);
            throw e;
        }
    }

    public Schema getSchema() {
        return schema;
    }

    public int numRowGroups() {
        return NativeLib.nativeReaderNumRowGroups(handle);
    }

    public void project(String[] columns) {
        NativeLib.nativeReaderSetProjection(handle, columns);
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

    public int rowGroupNumRows(int rgIndex) {
        int result = NativeLib.nativeReaderRowGroupNumRows(handle, rgIndex);
        if (result < 0) {
            throw new RuntimeException("failed to get row group num rows for index " + rgIndex);
        }
        return result;
    }

    /**
     * Returns column statistics for the given row group, keyed by column name.
     */
    public Map<String, ColumnStatistics> getRowGroupStatistics(int rgIndex) {
        String[] names = NativeLib.nativeReaderRowGroupStatNames(handle, rgIndex);
        if (names == null || names.length == 0) {
            return Collections.emptyMap();
        }
        long[] nullCounts = NativeLib.nativeReaderRowGroupStatNullCounts(handle, rgIndex);
        byte[][] mins = NativeLib.nativeReaderRowGroupStatMins(handle, rgIndex);
        byte[][] maxs = NativeLib.nativeReaderRowGroupStatMaxs(handle, rgIndex);
        Map<String, ColumnStatistics> result = new LinkedHashMap<>(names.length);
        for (int i = 0; i < names.length; i++) {
            result.put(names[i], new ColumnStatistics(nullCounts[i], mins[i], maxs[i]));
        }
        return Collections.unmodifiableMap(result);
    }

    public boolean bloomMightContain(int rgIndex, String columnName, Object value) {
        List<Field> fields = schema.getFields();
        int columnIndex = -1;
        ArrowType arrowType = null;
        for (int i = 0; i < fields.size(); i++) {
            if (fields.get(i).getName().equals(columnName)) {
                columnIndex = i;
                arrowType = fields.get(i).getType();
                break;
            }
        }
        if (columnIndex < 0) {
            throw new IllegalArgumentException("column not found: " + columnName);
        }
        int typeByte = arrowTypeByte(arrowType);
        byte[] encoded = encodeValue(typeByte, value);
        return NativeLib.nativeReaderBloomMightContain(
                handle, rgIndex, columnIndex, typeByte, encoded);
    }

    private static int arrowTypeByte(ArrowType type) {
        if (type instanceof ArrowType.Bool) return 0;
        if (type instanceof ArrowType.Int) {
            int bw = ((ArrowType.Int) type).getBitWidth();
            switch (bw) {
                case 8: return 1;
                case 16: return 2;
                case 32: return 3;
                case 64: return 4;
                default: throw new IllegalArgumentException("unsupported int width: " + bw);
            }
        }
        if (type instanceof ArrowType.FloatingPoint) {
            switch (((ArrowType.FloatingPoint) type).getPrecision()) {
                case SINGLE: return 5;
                case DOUBLE: return 6;
                default: throw new IllegalArgumentException("unsupported float precision");
            }
        }
        if (type instanceof ArrowType.Date) return 7;
        if (type instanceof ArrowType.Utf8) return 10;
        throw new IllegalArgumentException("unsupported arrow type for bloom: " + type);
    }

    private static byte[] encodeValue(int typeByte, Object value) {
        switch (typeByte) {
            case 0:
                return new byte[] { (byte) (((Boolean) value) ? 1 : 0) };
            case 1:
                return new byte[] { ((Number) value).byteValue() };
            case 2: {
                ByteBuffer bb = ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN);
                bb.putShort(((Number) value).shortValue());
                return bb.array();
            }
            case 3:
            case 7: {
                ByteBuffer bb = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
                bb.putInt(((Number) value).intValue());
                return bb.array();
            }
            case 5: {
                ByteBuffer bb = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
                bb.putFloat(((Number) value).floatValue());
                return bb.array();
            }
            case 4: {
                ByteBuffer bb = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
                bb.putLong(((Number) value).longValue());
                return bb.array();
            }
            case 6: {
                ByteBuffer bb = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
                bb.putDouble(((Number) value).doubleValue());
                return bb.array();
            }
            case 10:
                return ((String) value).getBytes(StandardCharsets.UTF_8);
            default:
                throw new IllegalArgumentException("unsupported type byte for value encoding: " + typeByte);
        }
    }

    @Override
    public void close() {
        if (handle != 0) {
            NativeLib.nativeReaderFree(handle);
            handle = 0;
        }
    }
}
