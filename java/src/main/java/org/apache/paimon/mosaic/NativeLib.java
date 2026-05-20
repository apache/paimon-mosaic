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

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

final class NativeLib {

    private static final String LIB_NAME = "mosaic_jni";

    static {
        loadNativeLibrary();
    }

    private NativeLib() {}

    private static void loadNativeLibrary() {
        // First try java.library.path (for development / manual override)
        try {
            System.loadLibrary(LIB_NAME);
            return;
        } catch (UnsatisfiedLinkError ignored) {
        }

        // Extract from JAR resources
        String os = normalizeOs(System.getProperty("os.name", ""));
        String arch = normalizeArch(System.getProperty("os.arch", ""));
        String libFileName = mapLibraryName(os);
        String resourcePath = "/native/" + os + "/" + arch + "/" + libFileName;

        try (InputStream in = NativeLib.class.getResourceAsStream(resourcePath)) {
            if (in == null) {
                throw new UnsatisfiedLinkError(
                        "Native library not found in JAR: " + resourcePath);
            }
            File tempFile = File.createTempFile("mosaic_jni", libFileName);
            tempFile.deleteOnExit();
            Files.copy(in, tempFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
            System.load(tempFile.getAbsolutePath());
        } catch (IOException e) {
            throw new UnsatisfiedLinkError(
                    "Failed to extract native library: " + e.getMessage());
        }
    }

    private static String normalizeOs(String osName) {
        String lower = osName.toLowerCase();
        if (lower.contains("linux")) {
            return "linux";
        } else if (lower.contains("mac") || lower.contains("darwin")) {
            return "macos";
        } else if (lower.contains("win")) {
            return "windows";
        }
        throw new UnsatisfiedLinkError("Unsupported OS: " + osName);
    }

    private static String normalizeArch(String archName) {
        String lower = archName.toLowerCase();
        if (lower.equals("amd64") || lower.equals("x86_64")) {
            return "x86_64";
        } else if (lower.equals("aarch64") || lower.equals("arm64")) {
            return "aarch64";
        }
        throw new UnsatisfiedLinkError("Unsupported architecture: " + archName);
    }

    private static String mapLibraryName(String os) {
        switch (os) {
            case "linux":
                return "libmosaic_jni.so";
            case "macos":
                return "libmosaic_jni.dylib";
            case "windows":
                return "mosaic_jni.dll";
            default:
                throw new UnsatisfiedLinkError("Unsupported OS: " + os);
        }
    }

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
    static native int nativeWriterNumRowGroups(long handle);
    static native int nativeWriterRowGroupNumStats(long handle, int rgIndex);
    static native int nativeWriterRowGroupStatColumnIndex(long handle, int rgIndex, int statIndex);
    static native long nativeWriterRowGroupStatNullCount(long handle, int rgIndex, int statIndex);
    static native byte[] nativeWriterRowGroupStatMin(long handle, int rgIndex, int statIndex);
    static native byte[] nativeWriterRowGroupStatMax(long handle, int rgIndex, int statIndex);

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
