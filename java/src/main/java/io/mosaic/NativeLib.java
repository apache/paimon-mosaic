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

import java.io.OutputStream;

final class NativeLib {

    static {
        System.loadLibrary("mosaic_jni");
    }

    private NativeLib() {}

    // Writer
    static native long nativeWriterOpen(OutputStream stream, long arrowSchemaAddr,
                                        int numBuckets, int compression, int zstdLevel,
                                        long rowGroupMaxSize, int maxDictTotalBytes,
                                        int maxDictEntries, int[] statsColumns,
                                        int pageSizeThreshold);
    static native void nativeWriterClose(long handle);
    static native void nativeWriterFree(long handle);
    static native long nativeWriterEstimatedSize(long handle);
    static native void nativeWriterWriteBatch(long writerHandle, long arrayAddr, long schemaAddr);

    // Reader
    static native long nativeReaderOpen(Object inputFile, long fileLength);
    static native void nativeReaderFree(long handle);
    static native int nativeReaderExportSchema(long handle, long schemaAddr);
    static native int nativeReaderNumRowGroups(long handle);
    static native long nativeReaderOpenRowGroup(long handle, int rgIndex);
    static native long nativeReaderOpenRowGroupProjected(long handle, int rgIndex, int[] columns);

    // RowGroupReader
    static native int nativeRowGroupReaderNumRows(long handle);
    static native int nativeRowGroupReaderReadColumns(long handle, long arrayAddr, long schemaAddr);
    static native void nativeRowGroupReaderFree(long handle);

    // Row group stats
    static native int nativeReaderRowGroupNumStats(long handle, int rgIndex);
    static native int nativeReaderRowGroupStatColumnIndex(long handle, int rgIndex, int statIndex);
    static native long nativeReaderRowGroupStatNullCount(long handle, int rgIndex, int statIndex);
    static native byte[] nativeReaderRowGroupStatMin(long handle, int rgIndex, int statIndex);
    static native byte[] nativeReaderRowGroupStatMax(long handle, int rgIndex, int statIndex);
}
