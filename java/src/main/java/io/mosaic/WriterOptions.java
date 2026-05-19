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

public class WriterOptions {

    public static final int COMPRESSION_ZSTD = 1;

    private int compression = COMPRESSION_ZSTD;
    private int zstdLevel = 1;
    private int numBuckets = 0;
    private long rowGroupMaxSize = 256L * 1024 * 1024;
    private int maxDictTotalBytes = 32 * 1024;
    private int maxDictEntries = 255;
    private int[] statsColumns = new int[0];
    private int pageSizeThreshold = 32 * 1024;

    public WriterOptions() {}

    public WriterOptions compression(int compression) {
        this.compression = compression;
        return this;
    }

    public WriterOptions zstdLevel(int level) {
        this.zstdLevel = level;
        return this;
    }

    public WriterOptions numBuckets(int numBuckets) {
        this.numBuckets = numBuckets;
        return this;
    }

    public WriterOptions rowGroupMaxSize(long size) {
        this.rowGroupMaxSize = size;
        return this;
    }

    public WriterOptions maxDictTotalBytes(int bytes) {
        this.maxDictTotalBytes = bytes;
        return this;
    }

    public WriterOptions maxDictEntries(int entries) {
        this.maxDictEntries = entries;
        return this;
    }

    public WriterOptions statsColumns(int... columns) {
        this.statsColumns = columns.clone();
        return this;
    }

    public WriterOptions pageSizeThreshold(int threshold) {
        this.pageSizeThreshold = threshold;
        return this;
    }

    int getCompression() { return compression; }
    int getZstdLevel() { return zstdLevel; }
    int getNumBuckets() { return numBuckets; }
    long getRowGroupMaxSize() { return rowGroupMaxSize; }
    int getMaxDictTotalBytes() { return maxDictTotalBytes; }
    int getMaxDictEntries() { return maxDictEntries; }
    int[] getStatsColumns() { return statsColumns; }
    int getPageSizeThreshold() { return pageSizeThreshold; }
}
